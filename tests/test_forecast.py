"""Tests for :mod:`occvent.forecast`: data layout, leakage, baselines, metrics."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from occvent.forecast import (
    FEATURE_COLUMNS,
    fill_short_gaps,
    LagBaseline,
    bootstrap_mae_by_day,
    build_models,
    daily_slot_table,
    make_features,
    predict_nonnegative,
    regression_metrics,
    slots_to_steps,
    period_mask,
)

TZ = "Australia/Melbourne"


def _minute_series(start_local: str, days: int, seed: int = 0) -> pd.Series:
    """Random integer counts on a 1-minute UTC grid covering ``days`` local days."""
    start = pd.Timestamp(start_local, tz=TZ).tz_convert("UTC")
    idx = pd.date_range(start, periods=days * 1440, freq="1min")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 6, idx.size).astype(float), index=idx)


def _table(days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-05-01", periods=days, freq="D").date
    return pd.DataFrame(rng.random((days, 96)) * 10, index=dates, columns=range(96))


# --------------------------------------------------------------- slot table

def test_daily_slot_table_shape_and_values():
    s = _minute_series("2025-05-05 00:00", 3)
    t = daily_slot_table(s)
    assert t.shape == (3, 96)
    local = s.index.tz_convert(TZ)
    first = s[(local.date == dt.date(2025, 5, 5)) & (local.hour == 9) & (local.minute < 15)]
    assert t.loc[dt.date(2025, 5, 5), 36] == pytest.approx(first.mean())       # 09:00-09:15 is slot 36
    assert np.isfinite(t.to_numpy()).all()


def test_daily_slot_table_daylight_saving_days():
    # Autumn change 6 Apr 2025 (03:00 -> 02:00): the 02:00 hour occurs twice and is averaged.
    t_autumn = daily_slot_table(_minute_series("2025-04-05 00:00", 3))
    assert np.isfinite(t_autumn.loc[dt.date(2025, 4, 6)].to_numpy()).all()
    # Spring change 5 Oct 2025 (02:00 -> 03:00): the 02:00 hour does not exist.
    t_spring = daily_slot_table(_minute_series("2025-10-04 00:00", 3))
    row = t_spring.loc[dt.date(2025, 10, 5)]
    assert row[[8, 9, 10, 11]].isna().all() and row.drop([8, 9, 10, 11]).notna().all()


def test_daily_slot_table_validates():
    with pytest.raises(ValueError):
        daily_slot_table(pd.Series([1.0, 2.0], index=pd.date_range("2025-01-01", periods=2, freq="1min")))
    with pytest.raises(ValueError):
        daily_slot_table(_minute_series("2025-05-05 00:00", 1), slot_minutes=7)


# ------------------------------------------------------------------ features

def test_features_columns_lags_and_calendar():
    t = _table()
    f = make_features(t)
    assert list(f.columns) == [*FEATURE_COLUMNS, "target"]
    assert f.index.get_level_values("date").min() == t.index[28]               # 4 weeks of history needed
    d, slot = t.index[40], 50
    row = f.loc[(d, slot)]
    assert row["target"] == t.loc[d, slot]
    assert row["lag_1d"] == t.loc[t.index[39], slot]
    assert row["lag_7d"] == t.loc[t.index[33], slot]
    assert row["lag_mean_4w"] == pytest.approx(np.mean([t.loc[t.index[40 - 7 * w], slot] for w in range(1, 5)]))
    assert row["prev_day_mean"] == pytest.approx(t.loc[t.index[39]].mean())
    assert row["prev_week_mean"] == pytest.approx(t.loc[t.index[33:40]].to_numpy().mean())
    assert row["lag_1d_2h"] == pytest.approx(t.loc[t.index[39], 46:54].mean())   # centred 9-slot window
    assert f.loc[(t.index[40], 0), "lag_1d_2h"] == pytest.approx(t.loc[t.index[39], 0:4].mean())  # stays within the day


def test_features_holiday_and_workday_flags():
    dates = pd.date_range("2025-03-20", "2025-05-10", freq="D").date
    t = pd.DataFrame(1.0, index=dates, columns=range(96))
    f = make_features(t).xs(0, level="slot")
    anzac, saturday, tuesday = dt.date(2025, 4, 25), dt.date(2025, 4, 26), dt.date(2025, 4, 29)
    assert (f.loc[anzac, "holiday"], f.loc[anzac, "workday"]) == (1.0, 0.0)
    assert (f.loc[saturday, "holiday"], f.loc[saturday, "workday"]) == (0.0, 0.0)
    assert (f.loc[tuesday, "holiday"], f.loc[tuesday, "workday"], f.loc[tuesday, "dow_1"]) == (0.0, 1.0, 1.0)


def test_features_have_no_leakage_from_the_target_day_or_later():
    t = _table()
    f0 = make_features(t)
    changed = t.copy()
    day = t.index[45]
    changed.loc[day] += 100.0                       # change everything measured on one day
    f1 = make_features(changed)
    feats = list(FEATURE_COLUMNS)
    upto = f0.index.get_level_values("date") <= day
    pd.testing.assert_frame_equal(f0.loc[upto, feats], f1.loc[upto, feats])   # features up to that day unaffected
    later = f0.index.get_level_values("date") == t.index[46]
    assert not np.allclose(f0.loc[later, "lag_1d"], f1.loc[later, "lag_1d"])  # but it does inform the next day


def test_missing_date_keeps_lags_aligned_to_calendar_days():
    t = _table().drop(index=_table().index[35])
    f = make_features(t)
    dates = set(f.index.get_level_values("date"))
    gap = _table().index[35]
    for k in (0, 1, 7, 14, 21, 28):                 # rows needing the missing day are dropped
        assert gap + dt.timedelta(days=k) not in dates
    row = f.loc[(_table().index[44], 10)]           # its history (days 16-43) does not include the gap
    assert row["lag_7d"] == t.loc[_table().index[37], 10]


def test_short_gaps_filled_for_lags_only():
    t = _table()
    day = t.index[40]
    t.loc[day, [8, 9, 10, 11]] = np.nan          # like the hour skipped at the spring clock change
    t.loc[t.index[41], list(range(20, 30))] = np.nan  # a 10-slot gap: too long to fill
    filled = fill_short_gaps(t)
    np.testing.assert_allclose(filled.loc[day, [8, 9, 10, 11]],
                               np.interp([8, 9, 10, 11], [7, 12], [t.loc[day, 7], t.loc[day, 12]]))
    assert filled.loc[t.index[41], list(range(20, 30))].isna().all()
    f = make_features(t)
    assert (day, 9) not in f.index                   # the target is never filled
    assert f.loc[(t.index[41], 9), "lag_1d"] == pytest.approx(filled.loc[day, 9])


def test_period_mask():
    f = make_features(_table())
    m = period_mask(f.index, "2025-06-01", "2025-06-10")
    d = pd.to_datetime(pd.Index(f.index.get_level_values("date")))
    assert m.sum() == 10 * 96 and d[m].min() == pd.Timestamp("2025-06-01")


# ------------------------------------------------------------- models, baselines

def test_lag_baseline_and_clone():
    f = make_features(_table())
    X, y = f[list(FEATURE_COLUMNS)], f["target"]
    b = clone(LagBaseline("lag_7d")).fit(X, y)
    np.testing.assert_array_equal(b.predict(X), X["lag_7d"].to_numpy())
    with pytest.raises(ValueError):
        LagBaseline("nope").fit(X, y)


def test_models_fit_and_predictions_are_nonnegative():
    f = make_features(_table(days=70))
    X, y = f[list(FEATURE_COLUMNS)], f["target"].to_numpy()
    models = build_models()
    assert set(models) == {"linear", "tree", "mlp"}
    for name in ("linear", "tree"):
        p = predict_nonnegative(models[name].fit(X, y), X)
        assert p.shape == y.shape and p.min() >= 0
    # A target that is an exact linear function of the features is recovered by linear regression.
    y_lin = 2.0 * X["lag_1d"].to_numpy() + 3.0 * X["workday"].to_numpy() + 1.0
    lin = build_models()["linear"].fit(X, y_lin)
    np.testing.assert_allclose(lin.predict(X), y_lin, atol=1e-8)


# ------------------------------------------------------------------ evaluation

def test_regression_metrics():
    m = regression_metrics([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 3.0, 2.0])
    assert m["MAE"] == pytest.approx(0.75)
    assert m["RMSE"] == pytest.approx(np.sqrt(5 / 4))
    assert m["bias"] == pytest.approx(-0.25)                        # under-predicts on average
    assert m["R2"] == pytest.approx(1 - 5 / 5)
    assert regression_metrics([1.0, 3.0], [1.0, 3.0])["R2"] == 1.0
    with pytest.raises(ValueError):
        regression_metrics([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        regression_metrics([1.0, np.nan], [1.0, 2.0])


def test_bootstrap_mae_by_day():
    dates = np.repeat(np.arange(30), 96)
    lo, hi = bootstrap_mae_by_day(np.full(dates.size, 0.7), dates, n_boot=200)
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)
    rng = np.random.default_rng(1)
    err = np.abs(rng.normal(0, 1, dates.size)) + np.repeat(rng.random(30), 96)
    lo, hi = bootstrap_mae_by_day(err, dates, n_boot=500)
    assert lo < err.mean() < hi
    with pytest.raises(ValueError):
        bootstrap_mae_by_day(err, dates[:-1])


def test_slots_to_steps_round_trip_conserves_person_time():
    s = _minute_series("2025-05-05 00:00", 3)
    f = daily_slot_table(s).stack()
    f.index = f.index.set_names(["date", "slot"])
    expanded = slots_to_steps(f, s.index)
    assert np.isfinite(expanded).all()
    assert expanded.sum() == pytest.approx(s.sum())                 # person-minutes conserved
    back = daily_slot_table(pd.Series(expanded, index=s.index))
    np.testing.assert_allclose(back.to_numpy(), daily_slot_table(s).to_numpy())
    with pytest.raises(ValueError):
        slots_to_steps(f, s.index.tz_localize(None))
