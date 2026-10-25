"""Tests for :mod:`occvent.co2model`: the exact solution checked against analytical results.

Each test compares the code with a result derived independently by hand:
steady state, exponential decay, linear growth with no ventilation, and
conservation of CO2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from occvent.co2model import (
    RoomParameters,
    co2_mass_balance,
    exact_solution,
    occupancy_from_grid,
    phi1,
    phi2,
    steady_state_ppm,
    step_mean_concentration,
)

P = RoomParameters(volume_m3=300.0, outdoor_co2_ppm=420.0, co2_generation_lps=0.0052)


def test_parameter_validation():
    for bad in ({"volume_m3": 0}, {"volume_m3": -5}, {"volume_m3": float("nan")},
                {"volume_m3": 100, "outdoor_co2_ppm": -1},
                {"volume_m3": 100, "co2_generation_lps": 0}):
        with pytest.raises(ValueError):
            RoomParameters(**bad)


def test_phi_functions_match_definitions_and_limits():
    x = np.array([0.0, 1e-12, 1e-6, 1e-3, 0.5, 2.0, 50.0])
    expected1 = np.array([1.0] + [(1 - np.exp(-v)) / v for v in x[1:]])
    np.testing.assert_allclose(phi1(x)[[0, 4, 5, 6]], expected1[[0, 4, 5, 6]], rtol=1e-14)
    np.testing.assert_allclose(phi1(x)[1:4], 1 - x[1:4] / 2, rtol=1e-6)
    # phi2: series branch and direct branch agree across the switch at 1e-3.
    np.testing.assert_allclose(phi2(np.array([0.0])), [0.5])
    left, right = phi2(np.array([1e-3 * (1 - 1e-9)])), phi2(np.array([1e-3]))
    np.testing.assert_allclose(left, right, rtol=1e-12)
    np.testing.assert_allclose(phi2(np.array([2.0])), [(2 - 1 + np.exp(-2)) / 4], rtol=1e-14)


def test_steady_state_formula():
    # 10 L/s per person with G = 0.0052 L/s gives 520 ppm above outdoor air.
    assert steady_state_ppm(20, 200.0, P) == pytest.approx(420 + 520)
    with pytest.raises(ValueError):
        steady_state_ppm(1, 0.0, P)


def test_constant_occupancy_approaches_steady_state_analytically():
    n_people, q, dt = 25.0, 150.0, 60.0
    n = 600
    c = exact_solution(np.full(n, n_people), q, P, dt)
    css = steady_state_ppm(n_people, q, P)
    k = q / 1000 / P.volume_m3
    t = np.arange(n + 1) * dt
    analytic = css + (P.outdoor_co2_ppm - css) * np.exp(-k * t)
    np.testing.assert_allclose(c, analytic, rtol=1e-12)


def test_empty_room_decays_exponentially():
    c0, q, dt = 1500.0, 90.0, 30.0
    c = exact_solution(np.zeros(200), q, P, dt, c0_ppm=c0)
    k = q / 1000 / P.volume_m3
    t = np.arange(201) * dt
    np.testing.assert_allclose(c, 420 + (c0 - 420) * np.exp(-k * t), rtol=1e-12)


def test_no_ventilation_gives_linear_growth():
    n_people, dt = 10.0, 60.0
    c = exact_solution(np.full(120, n_people), 0.0, P, dt, c0_ppm=500.0)
    rate = 1e6 * (0.0052 / 1000) * n_people / P.volume_m3  # ppm/s
    np.testing.assert_allclose(c, 500.0 + rate * np.arange(121) * dt, rtol=1e-12)


def test_loop_and_lfilter_agree():
    rng = np.random.default_rng(1)
    occ = rng.integers(0, 30, 5000).astype(float)
    c_loop = exact_solution(occ, 120.0, P, 60.0, method="loop")
    c_fast = exact_solution(occ, 120.0, P, 60.0, method="lfilter")
    np.testing.assert_allclose(c_fast, c_loop, rtol=1e-12)


def test_time_varying_ventilation_matches_piecewise_analytic():
    # Two phases: high then low ventilation with constant occupancy.
    occ = np.full(20, 12.0)
    q = np.array([200.0] * 10 + [50.0] * 10)
    dt = 120.0
    c = exact_solution(occ, q, P, dt)  # auto -> loop
    css1, css2 = steady_state_ppm(12, 200, P), steady_state_ppm(12, 50, P)
    k1, k2 = 0.2 / P.volume_m3, 0.05 / P.volume_m3
    c10 = css1 + (420 - css1) * np.exp(-k1 * 10 * dt)
    c20 = css2 + (c10 - css2) * np.exp(-k2 * 10 * dt)
    assert c[10] == pytest.approx(c10, rel=1e-12)
    assert c[20] == pytest.approx(c20, rel=1e-12)
    with pytest.raises(ValueError, match="constant"):
        exact_solution(occ, q, P, dt, method="lfilter")


def test_solution_stays_within_physical_bounds():
    rng = np.random.default_rng(7)
    occ = rng.integers(0, 40, 3000).astype(float)
    q = 250.0
    c = exact_solution(occ, q, P, 60.0)
    upper = steady_state_ppm(occ.max(), q, P)
    assert np.all(c >= P.outdoor_co2_ppm - 1e-9)
    assert np.all(c <= upper + 1e-9)


def test_step_mean_is_exact_time_average():
    # Constant N, Q: mean over one step = Css + (C0 - Css) (1 - e^{-kh}) / (kh).
    c = exact_solution(np.array([20.0]), 100.0, P, 600.0, c0_ppm=800.0)
    mean = step_mean_concentration(c, np.array([20.0]), 100.0, P, 600.0)
    css, kh = steady_state_ppm(20, 100, P), 0.1 / P.volume_m3 * 600
    assert mean[0] == pytest.approx(css + (800 - css) * (1 - np.exp(-kh)) / kh, rel=1e-13)
    # Compare with fine numerical quadrature of the analytic curve.
    t = np.linspace(0, 600, 200001)
    curve = css + (800 - css) * np.exp(-0.1 / P.volume_m3 * t)
    assert mean[0] == pytest.approx(np.trapezoid(curve, t) / 600, rel=1e-9)


def test_mass_balance_closes_to_round_off():
    rng = np.random.default_rng(3)
    occ = rng.integers(0, 25, 10_000) * rng.integers(0, 2, 10_000)
    q = rng.choice([0.0, 80.0, 300.0], 10_000)  # includes no-ventilation steps
    c = exact_solution(occ, q, P, 60.0, c0_ppm=600.0)
    mb = co2_mass_balance(c, occ, q, P, 60.0)
    assert mb["generated_m3"] > 0
    assert abs(mb["relative_residual"]) < 1e-10


def test_input_validation():
    with pytest.raises(ValueError):
        exact_solution(np.array([]), 100.0, P, 60.0)
    with pytest.raises(ValueError):
        exact_solution(np.array([1.0, np.nan]), 100.0, P, 60.0)
    with pytest.raises(ValueError):
        exact_solution(np.array([1.0, -1.0]), 100.0, P, 60.0)
    with pytest.raises(ValueError):
        exact_solution(np.array([1.0, 2.0]), np.array([100.0]), P, 60.0)
    with pytest.raises(ValueError):
        exact_solution(np.array([1.0]), -5.0, P, 60.0)
    with pytest.raises(ValueError):
        exact_solution(np.array([1.0]), 100.0, P, 0.0)
    with pytest.raises(ValueError):
        exact_solution(np.array([1.0]), 100.0, P, 60.0, c0_ppm=-1)
    with pytest.raises(ValueError, match="unknown method"):
        exact_solution(np.array([1.0]), 100.0, P, 60.0, method="magic")


def _grid(values):
    idx = pd.date_range("2024-05-01", periods=len(values), freq="1min", tz="UTC")
    return pd.DataFrame({"headcount": values}, index=idx)


def test_occupancy_from_grid_fill_options():
    g = _grid([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="no occupancy"):
        occupancy_from_grid(g)
    occ, dt, nf = occupancy_from_grid(g, fill="zero")
    assert dt == 60.0 and nf == 1
    np.testing.assert_array_equal(occ, [1.0, 0.0, 3.0])
    occ, _, _ = occupancy_from_grid(g, fill="ffill")
    np.testing.assert_array_equal(occ, [1.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="start"):
        occupancy_from_grid(_grid([np.nan, 1.0]), fill="ffill")
    with pytest.raises(ValueError, match="unknown fill"):
        occupancy_from_grid(_grid([1.0, 2.0]), fill="mean")


def test_occupancy_from_grid_rejects_irregular_index():
    idx = pd.DatetimeIndex(["2024-05-01 00:00", "2024-05-01 00:01", "2024-05-01 00:03"], tz="UTC")
    with pytest.raises(ValueError, match="regularly"):
        occupancy_from_grid(pd.DataFrame({"headcount": [1.0, 2.0, 3.0]}, index=idx))
