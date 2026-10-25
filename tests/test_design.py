"""Tests for :mod:`occvent.design`, checked against analytical results."""

from __future__ import annotations

import math

import numpy as np
import pytest

from occvent.co2model import RoomParameters, steady_state_ppm
from occvent.design import (
    RHO_CP_AIR_KJ_PER_M3K,
    bisection,
    dcv_ventilation,
    evaluate_ventilation,
    find_min_dcv_rate,
    find_min_ventilation,
    outdoor_air_energy,
    secant_bracketed,
)

P = RoomParameters(volume_m3=100.0)   # small room: short time constant, fast tests
DT = 60.0


# ---------------------------------------------------------------- root finders

def test_root_finders_on_sqrt2():
    f = lambda x: x * x - 2.0
    for finder in (bisection, secant_bracketed):
        r = finder(f, 0.0, 2.0, x_tol=1e-12, f_tol=1e-12)
        assert r.converged
        assert r.root == pytest.approx(math.sqrt(2), abs=1e-9)
        assert r.evaluations == len(r.history)


def test_secant_needs_fewer_evaluations_than_bisection():
    f = lambda x: x**3 - 2 * x - 5          # classic test function, root about 2.0946
    rb = bisection(f, 2.0, 3.0, x_tol=1e-10, f_tol=1e-10)
    rs = secant_bracketed(f, 2.0, 3.0, x_tol=1e-10, f_tol=1e-10)
    assert rs.root == pytest.approx(rb.root, abs=1e-8)
    assert rs.evaluations < rb.evaluations / 2


def test_secant_safeguard_keeps_iterates_in_bracket():
    # atan is flat far from 0, so raw secant steps can overshoot wildly.
    r = secant_bracketed(math.atan, -10.0, 20.0, x_tol=1e-10, f_tol=1e-12)
    assert r.converged and abs(r.root) < 1e-8
    assert all(-10.0 <= x <= 20.0 for x, _ in r.history)


def test_root_finder_bracket_validation_and_exact_ends():
    with pytest.raises(ValueError, match="same sign"):
        bisection(lambda x: x * x + 1, -1.0, 1.0)
    with pytest.raises(ValueError):
        secant_bracketed(lambda x: x, 1.0, 1.0)
    r = bisection(lambda x: x - 1.0, 1.0, 3.0)
    assert r.root == 1.0 and r.evaluations == 2


# ------------------------------------------------- minimum ventilation rates

def test_min_ventilation_8h_matches_steady_state_formula():
    # Constant occupancy for 3 days: the worst 8-h mean equals the steady state,
    # so Q_min = 1e6 * G * N / (850 - C_out).
    n_people = 12.0
    occ = np.full(3 * 1440, n_people)
    expected = 1e6 * P.co2_generation_lps * n_people / (850.0 - P.outdoor_co2_ppm)
    for method in ("secant", "bisection"):
        r = find_min_ventilation(occ, P, DT, criterion="8h", method=method, x_tol=1e-4, f_tol=1e-4)
        assert r.converged
        assert r.root == pytest.approx(expected, rel=1e-4)


def test_min_ventilation_peak_criterion():
    n_people = 30.0
    occ = np.concatenate([np.zeros(60), np.full(1440, n_people), np.zeros(600)])
    expected = 1e6 * P.co2_generation_lps * n_people / (1000.0 - P.outdoor_co2_ppm)
    r = find_min_ventilation(occ, P, DT, criterion="peak", x_tol=1e-4, f_tol=1e-4)
    assert r.root == pytest.approx(expected, rel=1e-4)
    # At Q_min the steady state sits on the limit.
    assert steady_state_ppm(n_people, r.root, P) == pytest.approx(1000.0, abs=0.05)


def test_min_ventilation_is_tight():
    occ = np.full(2 * 1440, 8.0)
    r = find_min_ventilation(occ, P, DT, x_tol=1e-3, f_tol=1e-3)
    assert evaluate_ventilation(occ, r.root * 1.01, P, DT)["max_8h_mean_ppm"] < 850
    assert evaluate_ventilation(occ, r.root * 0.99, P, DT)["max_8h_mean_ppm"] > 850


def test_min_ventilation_expands_bracket_and_validates():
    occ = np.full(1440, 50.0)
    r = find_min_ventilation(occ, P, DT, bracket_lps=(1.0, 2.0))   # needs about 605 L/s
    assert r.converged and r.root > 2.0
    with pytest.raises(ValueError, match="already met"):
        find_min_ventilation(np.zeros(1440), P, DT)                # empty room: met at any Q
    with pytest.raises(ValueError):
        find_min_ventilation(occ, P, DT, criterion="hourly")
    with pytest.raises(ValueError):
        find_min_ventilation(occ, P, DT, method="newton")


# ------------------------------------------------------------------------ DCV

def test_dcv_ventilation_rule():
    q = dcv_ventilation(np.array([0.0, 2.0, 10.0]), 12.0, 30.0)
    np.testing.assert_allclose(q, [30.0, 30.0, 120.0])
    with pytest.raises(ValueError):
        dcv_ventilation(np.ones(2), -1.0, 0.0)


def test_min_dcv_rate_matches_per_person_formula():
    # With q_floor = 0 and constant occupancy, C_ss = C_out + 1e6 G / q_p, independent of N.
    occ = np.full(3 * 1440, 7.0)
    r = find_min_dcv_rate(occ, P, DT, q_floor_lps=0.0, x_tol=1e-6, f_tol=1e-5)
    expected = 1e6 * P.co2_generation_lps / (850.0 - P.outdoor_co2_ppm)
    assert r.root == pytest.approx(expected, rel=1e-4)


def test_dcv_handles_empty_periods_with_zero_floor():
    # Q = 0 while empty: CO2 is trapped (no decay) but stays finite and >= outdoor level.
    occ = np.concatenate([np.full(600, 10.0), np.zeros(600)])
    m = evaluate_ventilation(occ, dcv_ventilation(occ, 15.0, 0.0), P, DT)
    assert np.isfinite(m["peak_ppm"]) and m["mean_ppm"] >= P.outdoor_co2_ppm


# --------------------------------------------------------------------- energy

def test_outdoor_air_energy_hand_calculation():
    e = outdoor_air_energy(100.0, 60.0, 60, delta_t_k=10.0)        # 100 L/s for 1 h
    assert e["air_volume_m3"] == pytest.approx(360.0)
    assert e["energy_kwh"] == pytest.approx(360.0 * RHO_CP_AIR_KJ_PER_M3K * 10.0 / 3600.0)
    e2 = outdoor_air_energy(np.array([0.0, 100.0]), 3600.0, 2, delta_t_k=10.0)
    assert e2["air_volume_m3"] == pytest.approx(360.0)
    with pytest.raises(ValueError):
        outdoor_air_energy(np.array([1.0, 2.0]), 60.0, 3, 5.0)
    with pytest.raises(ValueError):
        outdoor_air_energy(-1.0, 60.0, 3, 5.0)


# ------------------------------------------------- additions: 1/Q variable, cap

def test_inverse_variable_gives_same_root_with_fewer_secant_solves():
    rng = np.random.default_rng(0)
    occ = rng.integers(0, 25, 3 * 1440).astype(float)
    direct = find_min_ventilation(occ, P, DT, method="secant", variable="direct",
                                  bracket_lps=(20.0, 2000.0), x_tol=1e-3, f_tol=1e-3)
    inverse = find_min_ventilation(occ, P, DT, method="secant", variable="inverse",
                                   bracket_lps=(20.0, 2000.0), x_tol=1e-3, f_tol=1e-3)
    assert inverse.root == pytest.approx(direct.root, rel=1e-4)
    assert inverse.evaluations <= direct.evaluations
    assert inverse.method == "secant (1/x)"
    assert all(20.0 <= q <= 2000.0 for q, _ in inverse.history)   # history is reported in Q
    with pytest.raises(ValueError):
        find_min_ventilation(occ, P, DT, variable="log")


def test_dcv_cap():
    q = dcv_ventilation(np.array([0.0, 5.0, 100.0]), 10.0, 20.0, q_cap_lps=300.0)
    np.testing.assert_allclose(q, [20.0, 50.0, 300.0])
    with pytest.raises(ValueError):
        dcv_ventilation(np.ones(2), 10.0, 50.0, q_cap_lps=40.0)
    # A cap far above any demand changes nothing.
    occ = np.full(2 * 1440, 6.0)
    a = find_min_dcv_rate(occ, P, DT, q_floor_lps=0.0, x_tol=1e-6, f_tol=1e-5)
    b = find_min_dcv_rate(occ, P, DT, q_floor_lps=0.0, q_cap_lps=1e6, x_tol=1e-6, f_tol=1e-5)
    assert a.root == pytest.approx(b.root, rel=1e-9)
    # A cap too small to ever meet the limit is reported, not silently ignored.
    with pytest.raises(ValueError, match="upper bound"):
        find_min_dcv_rate(np.full(1440, 40.0), P, DT, q_floor_lps=0.0, q_cap_lps=50.0)
