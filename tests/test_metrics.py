"""Tests for :mod:`occvent.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from occvent.metrics import compliance_metrics, rolling_mean


def test_rolling_mean_matches_direct_average():
    rng = np.random.default_rng(0)
    v = rng.uniform(400, 1200, 1000)
    r = rolling_mean(v, 37)
    assert np.isnan(r[:36]).all()
    expected = np.array([v[i - 36:i + 1].mean() for i in range(36, 1000)])
    np.testing.assert_allclose(r[36:], expected, rtol=1e-12)


def test_rolling_mean_window_one_and_full_length():
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(rolling_mean(v, 1), v)
    r = rolling_mean(v, 3)
    assert np.isnan(r[:2]).all() and r[2] == pytest.approx(2.0)


def test_rolling_mean_precision_on_long_series():
    # Constant 850.1 ppm over 1.1 M steps: no drift from the cumulative sum.
    v = np.full(1_100_000, 850.1)
    r = rolling_mean(v, 480)
    assert np.nanmax(np.abs(r[479:] - 850.1)) < 1e-9


def test_rolling_mean_validation():
    with pytest.raises(ValueError):
        rolling_mean(np.ones(5), 0)
    with pytest.raises(ValueError):
        rolling_mean(np.ones(5), 6)
    with pytest.raises(ValueError):
        rolling_mean(np.array([1.0, np.nan]), 1)
    with pytest.raises(ValueError):
        rolling_mean(np.ones((2, 2)), 1)


def test_compliance_metrics_hand_example():
    # 24 h at 1-min steps: 600 ppm, except 2 h at 1100 ppm in the middle.
    v = np.full(1440, 600.0)
    v[600:720] = 1100.0
    m = compliance_metrics(v, 60.0)
    # Worst 8-h window contains the whole 2-h excursion: (6*600 + 2*1100)/8 = 725.
    assert m["max_8h_mean_ppm"] == pytest.approx(725.0)
    assert m["exceeds_8h_limit"] == 0.0
    assert m["margin_8h_ppm"] == pytest.approx(125.0)
    assert m["peak_ppm"] == 1100.0
    assert m["fraction_above_peak_limit"] == pytest.approx(120 / 1440)
    assert m["mean_ppm"] == pytest.approx((1320 * 600 + 120 * 1100) / 1440)


def test_compliance_metrics_detects_exceedance():
    m = compliance_metrics(np.full(600, 900.0), 60.0)
    assert m["exceeds_8h_limit"] == 1.0
    assert m["margin_8h_ppm"] == pytest.approx(-50.0)


def test_compliance_metrics_validation():
    with pytest.raises(ValueError, match="at least 8 h"):
        compliance_metrics(np.full(100, 500.0), 60.0)        # only 100 min
    with pytest.raises(ValueError, match="divide"):
        compliance_metrics(np.full(1000, 500.0), 7.0)        # 7 s does not divide 8 h
