"""Verification of the numerical solvers in :mod:`occvent.solvers`.

* **Convergence:** halving the step must cut the error by about 2^p, where
  p is the theoretical order (1, 1 and 4).
* **Stability:** explicit Euler must blow up for ``k h > 2``; implicit
  Euler must stay bounded and non-oscillatory for any ``h``.
* **Benchmark:** SciPy's adaptive solver must agree with the exact
  solution within its tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from occvent.co2model import RoomParameters, exact_solution, steady_state_ppm
from occvent.solvers import (
    SOLVERS,
    THEORETICAL_ORDER,
    explicit_euler,
    implicit_euler,
    rk4,
    solve_scipy,
)

# A small room with strong ventilation makes k*dt large (0.5), so solver
# errors are well above round-off and the convergence orders are clear.
P = RoomParameters(volume_m3=100.0)
Q = 833.3333333333334        # L/s -> k = Q/1000/V = 1/120 s^-1
DT = 60.0                    # k * DT = 0.5


def _random_occupancy(seed=0, n=300):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 20, n) * rng.integers(0, 2, n)).astype(float)


def _max_error(solver, occ, substeps):
    ref = exact_solution(occ, Q, P, DT, c0_ppm=700.0)
    return np.max(np.abs(solver(occ, Q, P, DT, substeps=substeps, c0_ppm=700.0) - ref))


@pytest.mark.parametrize("name", ["explicit_euler", "implicit_euler", "rk4"])
def test_observed_order_matches_theory(name):
    occ = _random_occupancy()
    m_values = [1, 2, 4, 8, 16]
    errors = np.array([_max_error(SOLVERS[name], occ, m) for m in m_values])
    assert np.all(np.diff(errors) < 0), f"error must decrease as h decreases: {errors}"
    orders = np.log2(errors[:-1] / errors[1:])
    p = THEORETICAL_ORDER[name]
    # The asymptotic order is reached on the finer steps.
    assert orders[-1] == pytest.approx(p, abs=0.15), f"{name}: orders {orders}"


def test_rk4_far_more_accurate_than_euler_at_same_step():
    occ = _random_occupancy(1)
    e_eu = _max_error(explicit_euler, occ, 4)
    e_ie = _max_error(implicit_euler, occ, 4)
    e_rk = _max_error(rk4, occ, 4)
    assert e_rk < 1e-3 * min(e_eu, e_ie)


def test_solvers_converge_to_exact_solution():
    occ = _random_occupancy(2)
    ref = exact_solution(occ, Q, P, DT, c0_ppm=700.0)
    for name in ("explicit_euler", "implicit_euler"):
        c = SOLVERS[name](occ, Q, P, DT, substeps=2000, c0_ppm=700.0)
        np.testing.assert_allclose(c, ref, rtol=1e-4)
    # RK4 local error ~ (k h)^5 / 120 per step: with k h = 0.5/8 that is ~8e-9 relative.
    np.testing.assert_allclose(rk4(occ, Q, P, DT, substeps=8, c0_ppm=700.0), ref, rtol=1e-7)


def test_constant_steady_state_is_preserved_by_every_solver():
    # Starting at the steady state, a consistent solver must stay there.
    css = steady_state_ppm(10, Q, P)
    occ = np.full(50, 10.0)
    for name in ("explicit_euler", "implicit_euler", "rk4"):
        c = SOLVERS[name](occ, Q, P, DT, c0_ppm=css)
        np.testing.assert_allclose(c, css, rtol=1e-12)


def test_explicit_euler_unstable_beyond_kh_equals_2():
    occ = np.zeros(60)
    # k*h = 2.5 with one sub-step: error multiplied by |1 - 2.5| = 1.5 per step.
    big_dt = 2.5 * 120.0
    c = explicit_euler(occ, Q, P, big_dt, c0_ppm=1000.0)
    dev = np.abs(c - P.outdoor_co2_ppm)
    assert dev[-1] > 1e6 * dev[0]
    assert np.all(np.sign(np.diff(c))[:-1] != np.sign(np.diff(c))[1:])  # oscillates
    # Just inside the limit (k*h = 1.9) it decays, though it still oscillates.
    c_ok = explicit_euler(occ, Q, P, 1.9 * 120.0, c0_ppm=1000.0)
    assert abs(c_ok[-1] - P.outdoor_co2_ppm) < abs(c_ok[0] - P.outdoor_co2_ppm)


def test_implicit_euler_stable_and_monotone_for_huge_steps():
    occ = np.zeros(60)
    c = implicit_euler(occ, Q, P, 100 * 120.0, c0_ppm=1000.0)  # k*h = 100
    assert np.all(np.diff(c) <= 0)  # monotone decay, no overshoot
    assert np.all(c >= P.outdoor_co2_ppm)


def test_scipy_benchmark_agrees_with_exact():
    occ = _random_occupancy(4, n=120)
    ref = exact_solution(occ, Q, P, DT, c0_ppm=700.0)
    c, info = solve_scipy(occ, Q, P, DT, c0_ppm=700.0, rtol=1e-10, atol=1e-8, full_output=True)
    np.testing.assert_allclose(c, ref, rtol=1e-7)
    assert info["nfev"] > 0
    c_lsoda = solve_scipy(occ, Q, P, DT, c0_ppm=700.0, method="LSODA", rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(c_lsoda, ref, rtol=1e-6)


def test_all_solvers_handle_time_varying_ventilation():
    occ = _random_occupancy(5, n=200)
    q = np.where(np.arange(200) < 100, Q, Q / 4)
    ref = exact_solution(occ, q, P, DT, c0_ppm=700.0)
    np.testing.assert_allclose(rk4(occ, q, P, DT, substeps=16, c0_ppm=700.0), ref, rtol=1e-8)
    np.testing.assert_allclose(implicit_euler(occ, q, P, DT, substeps=4000, c0_ppm=700.0), ref, rtol=1e-4)


def test_solver_argument_validation():
    occ = np.ones(5)
    for solver in (explicit_euler, implicit_euler, rk4):
        for bad in (0, -1, 1.5):
            with pytest.raises(ValueError, match="substeps"):
                solver(occ, Q, P, DT, substeps=bad)
        with pytest.raises(ValueError):
            solver(np.array([1.0, np.nan]), Q, P, DT)
        with pytest.raises(ValueError):
            solver(occ, Q, P, DT, c0_ppm=float("inf"))
