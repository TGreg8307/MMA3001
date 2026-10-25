"""Tests for :mod:`occvent.verification`."""

from __future__ import annotations

import numpy as np
import pytest

from occvent.co2model import RoomParameters
from occvent.solvers import explicit_euler, implicit_euler, rk4
from occvent.verification import (
    BYTES_PER_STEP,
    FLOPS_PER_SUBSTEP,
    STABILITY_LIMIT,
    arithmetic_intensity,
    best_time,
    convergence_study,
    empirical_roofline,
    observed_orders,
    stability_function,
    stiffness_summary,
)

P = RoomParameters(volume_m3=100.0)
Q = 833.3333333333334  # k = 1/120 s^-1, so k*60 s = 0.5


def test_stability_functions_match_one_solver_step():
    # One step from C0 with zero occupancy and zero outdoor CO2 multiplies C by R(-kh).
    p0 = RoomParameters(volume_m3=100.0, outdoor_co2_ppm=0.0)
    for r in (0.1, 0.9, 1.7, 2.5):
        dt = r * 120.0
        for name, solver in (("explicit_euler", explicit_euler),
                             ("implicit_euler", implicit_euler), ("rk4", rk4)):
            c = solver(np.zeros(1), Q, p0, dt, c0_ppm=1000.0)
            assert c[1] / 1000.0 == pytest.approx(float(stability_function(name, r)), rel=1e-12)


def test_stability_limits():
    assert abs(stability_function("explicit_euler", 2.0)) == pytest.approx(1.0)
    assert stability_function("rk4", STABILITY_LIMIT["rk4"]) == pytest.approx(1.0, abs=1e-12)
    assert abs(stability_function("rk4", 2.7)) < 1 < abs(stability_function("rk4", 2.9))
    r = np.linspace(0, 1000, 10001)
    assert np.all(np.abs(stability_function("implicit_euler", r)) <= 1)
    np.testing.assert_allclose(stability_function("exact", [0.0, 1.0]), [1.0, np.exp(-1)])
    with pytest.raises(ValueError):
        stability_function("magic", 1.0)
    with pytest.raises(ValueError):
        stability_function("rk4", -1.0)


def test_observed_orders_exact_power_law():
    h = np.array([1.0, 0.5, 0.25])
    np.testing.assert_allclose(observed_orders(h, 3.0 * h**2), [2.0, 2.0])
    with pytest.raises(ValueError):
        observed_orders([1.0], [1.0])
    with pytest.raises(ValueError):
        observed_orders([1.0, 0.5], [1.0, 0.0])


def test_convergence_study_reports_theoretical_orders():
    rng = np.random.default_rng(0)
    occ = (rng.integers(0, 20, 200) * rng.integers(0, 2, 200)).astype(float)
    for name, p in (("explicit_euler", 1), ("implicit_euler", 1), ("rk4", 4)):
        df = convergence_study(name, occ, Q, P, 60.0, [1, 2, 4, 8, 16], c0_ppm=700.0)
        assert list(df["substeps"]) == [1, 2, 4, 8, 16]
        assert np.isnan(df["observed_order"].iloc[0])
        assert df["observed_order"].iloc[-1] == pytest.approx(p, abs=0.15)
        assert df["kh"].iloc[0] == pytest.approx(0.5)
        assert df["rhs_evaluations"].iloc[-1] == 200 * 16 * (4 if name == "rk4" else 1)
        assert (df["runtime_s"] >= 0).all()


def test_convergence_study_validation():
    occ = np.ones(10)
    with pytest.raises(ValueError):
        convergence_study("exact", occ, Q, P, 60.0, [1, 2])
    with pytest.raises(ValueError):
        convergence_study("rk4", occ, Q, P, 60.0, [2, 1])


def test_arithmetic_intensity():
    assert arithmetic_intensity("explicit_euler") == pytest.approx(4 / 24)
    assert arithmetic_intensity("rk4", 8) == pytest.approx(21 * 8 / BYTES_PER_STEP)
    assert set(FLOPS_PER_SUBSTEP) == {"explicit_euler", "implicit_euler", "rk4", "exact"}
    with pytest.raises(ValueError):
        arithmetic_intensity("magic")
    with pytest.raises(ValueError):
        arithmetic_intensity("rk4", 0)


def test_best_time_and_roofline_are_positive():
    assert best_time(lambda: sum(range(100)), repeat=3) > 0
    roof = empirical_roofline(matrix_n=128, copy_n=2**16, repeat=2)
    assert roof["peak_flops"] > 0 and roof["bandwidth_bytes_per_s"] > 0
    assert roof["machine_balance"] == pytest.approx(roof["peak_flops"] / roof["bandwidth_bytes_per_s"])
    with pytest.raises(ValueError):
        best_time(lambda: None, repeat=0)


def test_stiffness_summary():
    s = stiffness_summary(300.0, 600.0)
    assert s["time_constant_s"] == pytest.approx(2000.0)
    assert s["air_changes_per_hour"] == pytest.approx(1.8)
    assert s["k_dt"] == pytest.approx(0.03)
    assert s["euler_max_stable_step_s"] == pytest.approx(4000.0)
    assert s["stiffness_ratio"] == 1.0
    with pytest.raises(ValueError):
        stiffness_summary(0.0, 600.0)
