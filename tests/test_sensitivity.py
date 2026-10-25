"""Tests for :mod:`occvent.sensitivity`, checked against exact and analytical results."""

from __future__ import annotations

import numpy as np
import pytest

from occvent.co2model import RoomParameters
from occvent.design import evaluate_ventilation, find_min_ventilation
from occvent.sensitivity import (
    UNIT_ROOM,
    ResponseCurve,
    aggregate_occupancy,
    elasticities,
    oat_sensitivity,
    percentile_summary,
    sample_inputs,
    scaled_metric,
)

DT = 60.0


def _random_occupancy(days: int = 3, seed: int = 0) -> np.ndarray:
    """Busy daytime blocks with random counts, empty at night (piecewise constant)."""
    rng = np.random.default_rng(seed)
    occ = np.zeros(days * 1440)
    for d in range(days):
        start = d * 1440 + 8 * 60
        occ[start:start + 10 * 60] = np.repeat(rng.integers(0, 30, 10 * 6), 10)
    return occ


# ------------------------------------------------------------------ scaling law

@pytest.mark.parametrize("v, g, c_out, k, q", [(600.0, 0.0052, 420.0, 1.0, 200.0),
                                               (250.0, 0.0041, 455.0, 1.17, 90.0),
                                               (1150.0, 0.0060, 400.0, 0.83, 400.0)])
def test_scaling_law_is_exact(v, g, c_out, k, q):
    occ = _random_occupancy()
    full = evaluate_ventilation(k * occ, q, RoomParameters(v, c_out, g), DT)
    unit = evaluate_ventilation(occ, q / v, UNIT_ROOM, DT)
    for key in ("max_8h_mean_ppm", "peak_ppm", "mean_ppm"):
        assert full[key] == pytest.approx(float(scaled_metric(unit[key], v, g, c_out, k)), rel=1e-11)


def test_response_curve_constant_occupancy_matches_steady_state():
    # Constant N for 3 days: M_8h(a) = 1e6 N / a (a pure power law), so the
    # log-log interpolation is exact and Q_min = 1e6 G k N / (850 - C_out).
    n = 9.0
    rc = ResponseCurve.build(np.full(3 * 1440, n), DT)
    for v, g, c_out, k in [(600.0, 0.0052, 420.0, 1.0), (300.0, 0.0045, 450.0, 1.2)]:
        expected = 1e6 * g * k * n / (850.0 - c_out)
        assert float(rc.q_min("8h", v, g, c_out, k)) == pytest.approx(expected, rel=1e-6)


def test_response_curve_matches_full_model():
    occ = _random_occupancy(seed=3)
    rc = ResponseCurve.build(occ, DT, np.geomspace(0.05, 10.0, 161))
    for v, g, c_out, k in [(600.0, 0.0052, 420.0, 1.0), (400.0, 0.0047, 440.0, 1.1)]:
        p = RoomParameters(v, c_out, g)
        for crit in ("8h", "peak"):
            full = find_min_ventilation(k * occ, p, DT, criterion=crit, variable="inverse",
                                        bracket_lps=(10.0, 2000.0), x_tol=1e-5, f_tol=1e-5).root
            assert float(rc.q_min(crit, v, g, c_out, k)) == pytest.approx(full, rel=2e-3)
        m = evaluate_ventilation(k * occ, 300.0, p, DT)["max_8h_mean_ppm"]
        assert float(rc.metric("8h", 300.0, v, g, c_out, k)) == pytest.approx(m, rel=1e-3)


def test_generation_and_count_factor_enter_only_as_a_product():
    rc = ResponseCurve.build(_random_occupancy(seed=5), DT)
    a = rc.q_min("8h", 600.0, 0.008, 420.0, 1.0)
    b = rc.q_min("8h", 600.0, 0.004, 420.0, 2.0)
    assert float(a) == pytest.approx(float(b), rel=1e-12)


def test_response_curve_is_vectorised_and_validates():
    rc = ResponseCurve.build(_random_occupancy(), DT)
    q = rc.q_min("peak", np.array([400.0, 600.0, 900.0]), 0.0052, 420.0)
    assert q.shape == (3,) and np.all(np.diff(q) < 0)      # a larger room buffers peaks
    with pytest.raises(ValueError, match="exceed"):
        rc.q_min("8h", 600.0, 0.0052, 900.0)
    with pytest.raises(ValueError, match="tabulated range"):
        rc.q_min("8h", 600.0, 1e-6, 420.0)                  # needs an absurdly low flow
    with pytest.raises(ValueError, match="tabulated range"):
        rc.metric("8h", 1e6, 600.0, 0.0052, 420.0)
    with pytest.raises(ValueError):
        rc.q_min("hourly", 600.0, 0.0052, 420.0)
    with pytest.raises(ValueError):
        ResponseCurve.build(np.zeros(1440), DT)             # empty room: metric is zero
    with pytest.raises(ValueError):
        ResponseCurve.build(_random_occupancy(), DT, np.array([1.0, 0.5]))


# --------------------------------------------------------------------------- OAT

def test_oat_sensitivity_on_linear_function():
    table = oat_sensitivity(lambda x, y, z: x + 2 * y + 0 * z, {"x": 1.0, "y": 1.0, "z": 5.0},
                            {"x": (0.0, 2.0), "y": (0.5, 1.5), "z": (0.0, 10.0)})
    assert list(table["parameter"]) == ["x", "y", "z"]       # x and y tie (2.0); stable sort keeps order
    np.testing.assert_allclose(table["swing"], [2.0, 2.0, 0.0])
    assert table.loc[0, "f_nominal"] == 3.0
    row_y = table.set_index("parameter").loc["y"]
    assert (row_y["f_low"], row_y["f_high"]) == (2.0, 4.0)
    with pytest.raises(KeyError):
        oat_sensitivity(lambda x: x, {"x": 1.0}, {"w": (0.0, 1.0)})


def test_elasticities_of_power_law():
    f = lambda x, y, z: 3.0 * x**2 / y * np.sqrt(z)
    e = elasticities(f, {"x": 2.0, "y": 5.0, "z": 7.0})
    assert e["x"] == pytest.approx(2.0, abs=1e-3)
    assert e["y"] == pytest.approx(-1.0, abs=1e-3)
    assert e["z"] == pytest.approx(0.5, abs=1e-3)
    assert set(elasticities(f, {"x": 2.0, "y": 5.0, "z": 7.0}, names=["x"])) == {"x"}
    with pytest.raises(ValueError):
        elasticities(f, {"x": 0.0, "y": 5.0, "z": 7.0})
    with pytest.raises(ValueError):
        elasticities(f, {"x": 2.0, "y": 5.0, "z": 7.0}, rel_step=0.0)


# ------------------------------------------------------------------- Monte Carlo

def test_sample_inputs_bounds_reproducibility_and_shapes():
    spec = {"v": ("loguniform", 300.0, 1200.0), "g": ("triangular", 0.004, 0.0052, 0.006),
            "c": ("uniform", 400.0, 460.0), "k": ("normal", 1.0, 0.1)}
    s1, s2 = sample_inputs(spec, 20000, seed=7), sample_inputs(spec, 20000, seed=7)
    for name in spec:
        assert s1[name].shape == (20000,)
        np.testing.assert_array_equal(s1[name], s2[name])
    assert 300.0 <= s1["v"].min() and s1["v"].max() <= 1200.0
    assert np.median(s1["v"]) == pytest.approx(600.0, rel=0.03)    # geometric mean of 300 and 1200
    assert 0.004 <= s1["g"].min() and s1["g"].max() <= 0.006
    assert s1["g"].mean() == pytest.approx((0.004 + 0.0052 + 0.006) / 3, rel=0.01)
    assert s1["k"].std() == pytest.approx(0.1, rel=0.03)
    assert not np.array_equal(s1["c"], sample_inputs(spec, 20000, seed=8)["c"])


@pytest.mark.parametrize("spec", [{"x": ("beta", 1.0, 2.0)}, {"x": ("uniform", 1.0)},
                                  {"x": ("uniform", 2.0, 1.0)}, {"x": ("loguniform", 0.0, 1.0)},
                                  {"x": ("triangular", 1.0, 3.0, 2.0)}, {"x": ("normal", 1.0, 0.0)}])
def test_sample_inputs_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        sample_inputs(spec, 10)


def test_percentile_summary():
    v = np.arange(101.0)
    s = percentile_summary(v)
    assert (s["p5"], s["p50"], s["p95"]) == (5.0, 50.0, 95.0)
    assert s["mean"] == 50.0 and s["n"] == 101
    assert s["sem"] == pytest.approx(v.std(ddof=1) / np.sqrt(101))
    with pytest.raises(ValueError):
        percentile_summary(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        percentile_summary(np.array([]))


# --------------------------------------------------------- numerical sensitivity

def test_aggregate_occupancy_conserves_person_time():
    occ = np.arange(125, dtype=float)
    agg = aggregate_occupancy(occ, 5)
    assert agg.shape == (25,)
    assert agg.sum() * 5 == pytest.approx(occ.sum())
    np.testing.assert_allclose(agg[:2], [2.0, 7.0])
    assert aggregate_occupancy(occ, 60).shape == (2,)        # incomplete tail dropped
    np.testing.assert_array_equal(aggregate_occupancy(occ, 1), occ)
    with pytest.raises(ValueError):
        aggregate_occupancy(occ, 0)
    with pytest.raises(ValueError):
        aggregate_occupancy(occ, 2.5)
