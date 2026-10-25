"""Tests for :func:`occvent.cleaning.resample_occupancy`.

The key correctness property is that resampling is an *exact* time-average
of the zero-order-hold signal. It is checked two ways: against values
worked out by hand, and against a brute-force second-by-second evaluation
on random event sequences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helpers import make_events
from occvent.cleaning import ResampleSettings, resample_occupancy


def test_hand_worked_example():
    ev = make_events([
        ("2024-05-01 00:00:00", 2),
        ("2024-05-01 00:00:30", 4),
        ("2024-05-01 00:01:30", 0),
        ("2024-05-01 00:03:00", 1),
    ])
    grid, rep = resample_occupancy(ev)
    assert len(grid) == 3
    np.testing.assert_allclose(grid["headcount"], [3.0, 2.0, 0.0])
    np.testing.assert_allclose(grid["coverage"], [1.0, 1.0, 1.0])
    assert grid["valid"].all()
    assert grid.index[0] == pd.Timestamp("2024-05-01 00:00", tz="UTC")
    assert grid.index.name == "t_utc"
    # person-seconds: 2*30 + 4*60 + 0*90 = 300 s = 1/12 h
    assert rep["person_hours_observed"] == pytest.approx(300 / 3600)


def test_partial_first_bin_is_not_valid():
    ev = make_events([("2024-05-01 00:00:20", 3), ("2024-05-01 00:01:00", 0), ("2024-05-01 00:02:00", 0)])
    grid, _ = resample_occupancy(ev)
    assert grid["headcount"].iloc[0] == pytest.approx(3.0)
    assert grid["coverage"].iloc[0] == pytest.approx(40 / 60)
    assert not grid["valid"].iloc[0]
    assert grid["valid"].iloc[1]


def test_min_coverage_threshold():
    ev = make_events([("2024-05-01 00:00:20", 3), ("2024-05-01 00:02:00", 0)])
    grid, _ = resample_occupancy(ev, ResampleSettings(min_coverage=0.5))
    assert grid["valid"].iloc[0]  # 40/60 >= 0.5


def test_occupied_hold_is_truncated():
    ev = make_events([("2024-05-01 00:00", 5), ("2024-05-01 00:05", 0), ("2024-05-01 00:06", 0)])
    grid, rep = resample_occupancy(ev, ResampleSettings(max_occupied_hold="2min"))
    np.testing.assert_allclose(grid["headcount"].iloc[:2], [5.0, 5.0])
    assert grid["headcount"].iloc[2:5].isna().all()
    np.testing.assert_allclose(grid["coverage"], [1, 1, 0, 0, 0, 1])
    assert list(grid["valid"]) == [True, True, False, False, False, True]
    assert rep["occupied_holds_truncated"] == 1
    assert rep["vacant_holds_truncated"] == 0
    assert rep["unobserved_hours_from_truncation"] == pytest.approx(3 / 60)
    assert rep["longest_occupied_hold_hours"] == pytest.approx(5 / 60)


def test_long_vacant_hold_is_kept_by_default_but_can_be_limited():
    ev = make_events([("2024-05-01 00:00", 0), ("2024-05-03 02:00", 2), ("2024-05-03 02:01", 0)])
    grid, rep = resample_occupancy(ev)  # 50 h vacant < 96 h default
    assert grid["valid"].all()
    assert rep["vacant_holds_truncated"] == 0
    grid2, rep2 = resample_occupancy(ev, ResampleSettings(max_vacant_hold="24h"))
    assert rep2["vacant_holds_truncated"] == 1
    assert (~grid2["valid"]).sum() == 26 * 60  # 50 h - 24 h unobserved


def test_hour_step():
    ev = make_events([("2024-05-01 00:00", 6), ("2024-05-01 00:15", 2), ("2024-05-01 01:00", 0)])
    grid, _ = resample_occupancy(ev, ResampleSettings(step="1h"))
    assert len(grid) == 1
    assert grid["headcount"].iloc[0] == pytest.approx((6 * 15 + 2 * 45) / 60)


def test_single_event_on_bin_edge():
    grid, _ = resample_occupancy(make_events([("2024-05-01 00:00", 1)]))
    assert len(grid) == 1
    assert np.isnan(grid["headcount"].iloc[0])
    assert grid["coverage"].iloc[0] == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_brute_force_on_random_events(seed):
    """Compare with explicit per-second evaluation of the held signal."""
    rng = np.random.default_rng(seed)
    occ_hold, vac_hold, step = 300, 900, 60
    n_events = 200
    gaps = rng.integers(1, 1200, size=n_events - 1)
    secs = np.concatenate(([7], 7 + np.cumsum(gaps)))  # deliberately not on a bin edge
    counts = rng.integers(0, 6, size=n_events) * rng.integers(0, 2, size=n_events)
    t0 = pd.Timestamp("2024-06-01", tz="UTC")
    ev = pd.DataFrame({"t": t0 + pd.to_timedelta(secs, unit="s"), "headcount": counts})

    settings = ResampleSettings(step=f"{step}s", max_occupied_hold=f"{occ_hold}s",
                                max_vacant_hold=f"{vac_hold}s")
    grid, rep = resample_occupancy(ev, settings)

    # Brute force: value of the held signal in each 1-second slot (NaN = unobserved).
    total = int(np.ceil(secs[-1] / step) * step)
    signal = np.full(total, np.nan)
    for i in range(n_events - 1):
        limit = occ_hold if counts[i] > 0 else vac_hold
        stop = min(secs[i + 1], secs[i] + limit)
        signal[secs[i]:stop] = counts[i]
    per_bin = signal.reshape(-1, step)
    observed = (~np.isnan(per_bin)).sum(axis=1)
    sums = np.nansum(per_bin, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected_mean = np.where(observed > 0, sums / observed, np.nan)

    assert len(grid) == len(expected_mean)
    np.testing.assert_allclose(grid["coverage"], observed / step, rtol=0, atol=1e-12)
    np.testing.assert_allclose(grid["headcount"], expected_mean, rtol=1e-12, equal_nan=True)
    # Conservation of person-time.
    assert rep["person_hours_observed"] == pytest.approx(np.nansum(signal) / 3600, rel=1e-12)


def test_input_validation():
    good = make_events([("2024-05-01 00:00", 1), ("2024-05-01 00:01", 0)])
    with pytest.raises(ValueError, match="empty"):
        resample_occupancy(good.iloc[:0])
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_occupancy(good.iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_occupancy(pd.concat([good, good.iloc[[1]]], ignore_index=True))
    naive = good.assign(t=good["t"].dt.tz_localize(None))
    with pytest.raises(ValueError, match="tz-aware"):
        resample_occupancy(naive)
    with pytest.raises(ValueError, match="whole seconds"):
        resample_occupancy(good.assign(t=good["t"] + pd.Timedelta(milliseconds=500)))
    with pytest.raises(ValueError, match="non-negative"):
        resample_occupancy(good.assign(headcount=[-1, 0]))
    for bad in (ResampleSettings(step="1.5s"), ResampleSettings(step="0s"),
                ResampleSettings(max_occupied_hold="-1h"), ResampleSettings(step="banana"),
                ResampleSettings(min_coverage=0.0), ResampleSettings(min_coverage=1.5)):
        with pytest.raises(ValueError):
            resample_occupancy(good, bad)
