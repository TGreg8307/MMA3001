"""Ventilation design: compliance checks, minimum ventilation rates and energy.

This module answers the project's engineering questions:

1. **Compliance:** for a given outdoor-air supply ``Q`` (constant, or varying in
   time), what are the 8-hour mean CO2, the peak and the time above limits?
2. **Minimum ventilation rate:** the smallest constant ``Q`` meeting a CO2
   criterion. This is a scalar root-finding problem
   ``F(Q) = metric(C(t; Q)) - limit = 0``, where every evaluation of ``F``
   solves the CO2 ODE. It has the same structure as the **shooting method**
   (MMA3001 Notebook 9.4): guess a parameter, integrate the ODE, measure the
   miss, and update the guess.
3. **Demand-controlled ventilation (DCV):** supply that follows the measured
   occupancy, ``Q(t) = max(Q_floor, q_p * N(t))``, and the smallest
   per-person rate ``q_p`` that meets the criterion.
4. **Energy:** outdoor-air volume and the thermal energy needed to condition it.

Monotonicity (why bracketing root-finders are safe here)
--------------------------------------------------------
With ``C >= C_out``, raising ``Q`` at any instant can only lower ``C``. The
sensitivity ``S = dC/dQ`` obeys ``V dS/dt = (C_out - C) - Q S`` with
``S(0) = 0``, and its forcing ``C_out - C <= 0``, so ``S <= 0`` for all ``t``.
Every metric used here (8-h mean, peak) is therefore non-increasing in ``Q``
(and in ``q_p`` for DCV). ``F`` has at most one sign change, so bisection
always converges once a bracket is found.

Root finders are implemented from scratch (bisection; secant safeguarded by a
bracket) and recorded iteration by iteration, so their convergence can be
compared. SciPy's ``brentq`` serves as an independent cross-check in the notebook.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .co2model import RoomParameters, exact_solution, step_mean_concentration, validate_inputs
from .metrics import NCC_8H_LIMIT_PPM, PEAK_LIMIT_PPM, compliance_metrics

#: Volumetric heat capacity of air [kJ/(m3 K)]: rho = 1.2 kg/m3 times c_p = 1.005 kJ/(kg K).
RHO_CP_AIR_KJ_PER_M3K: float = 1.2 * 1.005

#: Criterion name -> (metric key in :func:`occvent.metrics.compliance_metrics`, default limit [ppm]).
CRITERIA: dict[str, tuple[str, float]] = {
    "8h": ("max_8h_mean_ppm", NCC_8H_LIMIT_PPM),
    "peak": ("peak_ppm", PEAK_LIMIT_PPM),
}


# ---------------------------------------------------------------------------
# Compliance of a ventilation scenario
# ---------------------------------------------------------------------------

def evaluate_ventilation(
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
    c0_ppm: float | None = None,
) -> dict[str, float]:
    """Simulate CO2 exactly for a ventilation scenario and return its compliance metrics.

    Parameters
    ----------
    occupancy : array_like
        People during each step (length ``n``, covering at least 8 hours).
    ventilation_lps : float or array_like
        Outdoor-air flow [L/s], constant or one value per step.
    params : RoomParameters
        Room volume, outdoor CO2 and generation rate.
    dt_s : float
        Step length [s].
    c0_ppm : float, optional
        Initial CO2. Defaults to the outdoor concentration.

    Returns
    -------
    dict
        The keys of :func:`occvent.metrics.compliance_metrics`.
    """
    c = exact_solution(occupancy, ventilation_lps, params, dt_s, c0_ppm)
    mean = step_mean_concentration(c, occupancy, ventilation_lps, params, dt_s)
    return compliance_metrics(mean, dt_s)


# ---------------------------------------------------------------------------
# Root finders (own implementations)
# ---------------------------------------------------------------------------

@dataclass
class RootResult:
    """Outcome of a root-finding run.

    Attributes
    ----------
    root : float
        Final estimate of ``x`` with ``f(x) = 0``.
    f_root : float
        ``f(root)``.
    converged : bool
        Whether a stopping tolerance was met within ``max_iter``.
    evaluations : int
        Number of calls to ``f``, including the two bracket ends. Each call is
        one full ODE solve.
    method : str
        ``"bisection"`` or ``"secant"``.
    history : list of (float, float)
        Every ``(x, f(x))`` evaluated, in order.
    """

    root: float
    f_root: float
    converged: bool
    evaluations: int
    method: str
    history: list[tuple[float, float]] = field(default_factory=list)


def _check_bracket(f: Callable[[float], float], lo: float, hi: float) -> tuple[float, float]:
    """Evaluate ``f`` at both ends and check that they have opposite signs."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        raise ValueError(f"need a finite bracket with lo < hi, got ({lo}, {hi})")
    f_lo, f_hi = float(f(lo)), float(f(hi))
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        raise ValueError("f is not finite at the bracket ends")
    if f_lo * f_hi > 0:
        raise ValueError(f"f(lo)={f_lo:.4g} and f(hi)={f_hi:.4g} have the same sign: the bracket does not contain a root")
    return f_lo, f_hi


def bisection(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    x_tol: float = 1e-2,
    f_tol: float = 1e-2,
    max_iter: int = 200,
) -> RootResult:
    """Find a root of ``f`` in ``[lo, hi]`` by bisection.

    Each iteration halves the bracket, so the error after ``n`` iterations is
    at most ``(hi - lo) / 2**n``. This is linear convergence with ratio 1/2,
    guaranteed for any continuous ``f`` that changes sign on the bracket.

    Parameters
    ----------
    f : callable
        Continuous scalar function.
    lo, hi : float
        Bracket ends with ``f(lo) * f(hi) <= 0``.
    x_tol : float, default 0.01
        Stop when the bracket is narrower than this (same units as ``x``).
    f_tol : float, default 0.01
        Stop when ``|f(x)| <= f_tol`` (same units as ``f``).
    max_iter : int, default 200
        Maximum number of bisection steps.

    Returns
    -------
    RootResult

    Raises
    ------
    ValueError
        If the bracket is invalid or does not contain a sign change.
    """
    f_lo, f_hi = _check_bracket(f, lo, hi)
    history = [(lo, f_lo), (hi, f_hi)]
    if f_lo == 0 or f_hi == 0:
        x = lo if f_lo == 0 else hi
        return RootResult(x, 0.0, True, 2, "bisection", history)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = float(f(mid))
        history.append((mid, f_mid))
        if abs(f_mid) <= f_tol or (hi - lo) <= x_tol:
            return RootResult(mid, f_mid, True, len(history), "bisection", history)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return RootResult(mid, f_mid, False, len(history), "bisection", history)


def secant_bracketed(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    x_tol: float = 1e-2,
    f_tol: float = 1e-2,
    max_iter: int = 100,
) -> RootResult:
    """Find a root of ``f`` in ``[lo, hi]`` by the secant method, safeguarded by bisection.

    The secant method is Newton's method with the derivative replaced by the
    slope through the last two iterates::

        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)

    It needs no derivative (useful here, because each ``f`` is a whole ODE
    solve) and converges superlinearly (order about 1.618) near a simple root.
    On its own it can jump out of the region where the root lies. A sign-
    change bracket is therefore kept: any secant step that would leave the
    bracket, or that cannot be computed, is replaced by a bisection step. This
    guarantees convergence.

    Parameters and return value are as for :func:`bisection`. Iteration also
    stops when successive iterates differ by less than ``x_tol``.
    """
    f_lo, f_hi = _check_bracket(f, lo, hi)
    history = [(lo, f_lo), (hi, f_hi)]
    if f_lo == 0 or f_hi == 0:
        x = lo if f_lo == 0 else hi
        return RootResult(x, 0.0, True, 2, "secant", history)
    a, fa, b, fb = lo, f_lo, hi, f_hi          # sign-change bracket
    x0, f0, x1, f1 = lo, f_lo, hi, f_hi        # last two iterates
    for _ in range(max_iter):
        denom = f1 - f0
        x2 = x1 - f1 * (x1 - x0) / denom if denom != 0 else np.nan
        if not (np.isfinite(x2) and a < x2 < b):
            x2 = 0.5 * (a + b)                 # safeguard: bisection step
        f2 = float(f(x2))
        history.append((x2, f2))
        if abs(f2) <= f_tol or abs(x2 - x1) <= x_tol or (b - a) <= x_tol:
            return RootResult(x2, f2, True, len(history), "secant", history)
        if np.sign(f2) == np.sign(fa):
            a, fa = x2, f2
        else:
            b, fb = x2, f2
        x0, f0, x1, f1 = x1, f1, x2, f2
    return RootResult(x2, f2, False, len(history), "secant", history)


ROOT_FINDERS: dict[str, Callable[..., RootResult]] = {
    "bisection": bisection,
    "secant": secant_bracketed,
}


def _solve_monotone_decreasing(
    F: Callable[[float], float],
    lo: float,
    hi: float,
    method: str,
    x_tol: float,
    f_tol: float,
    max_expand: int = 20,
    variable: str = "direct",
) -> RootResult:
    """Root of a non-increasing ``F``. Expands ``hi`` (doubling) until ``F(hi) <= 0``.

    With ``variable="inverse"`` the root finder works in ``u = 1/x`` instead of
    ``x``. CO2 above outdoor level scales roughly as ``1/Q`` (``C_ss - C_out =
    1e6 G N / Q``), so ``F`` is nearly *linear* in ``1/Q``. Secant steps are
    then almost exact. The returned ``root`` and ``history`` are always in
    terms of ``x``. ``x_tol`` applies to ``x``: it is converted to an
    equivalent tolerance on ``u`` near the upper end of the bracket.
    """
    if variable not in ("direct", "inverse"):
        raise ValueError(f"variable must be 'direct' or 'inverse', got {variable!r}")
    if method not in ROOT_FINDERS:
        raise ValueError(f"method must be one of {sorted(ROOT_FINDERS)}, got {method!r}")
    if lo <= 0 or hi <= lo:
        raise ValueError(f"need 0 < lo < hi, got ({lo}, {hi})")
    cache: dict[float, float] = {}
    F_raw = F

    def F(x: float) -> float:  # memoised: each ODE solve is done once per x
        if x not in cache:
            cache[x] = float(F_raw(x))
        return cache[x]

    if F(lo) <= 0:
        raise ValueError(f"the criterion is already met at the lower bound {lo}; lower it")
    for _ in range(max_expand):
        if F(hi) <= 0:
            break
        hi *= 2.0
    else:
        raise ValueError("could not find an upper bound that meets the criterion")
    if variable == "direct":
        return ROOT_FINDERS[method](F, lo, hi, x_tol=x_tol, f_tol=f_tol)
    # u = 1/x reverses the bracket order; F(1/u) is non-decreasing in u.
    u_tol = x_tol / hi**2          # |du| = |dx| / x^2, conservative at the largest x
    res = ROOT_FINDERS[method](lambda u: F(1.0 / u), 1.0 / hi, 1.0 / lo, x_tol=u_tol, f_tol=f_tol)
    res.root = 1.0 / res.root
    res.history = [(1.0 / u, f) for u, f in res.history]
    res.method = f"{res.method} (1/x)"
    return res


def _criterion(criterion: str, limit_ppm: float | None) -> tuple[str, float]:
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {sorted(CRITERIA)}, got {criterion!r}")
    key, default = CRITERIA[criterion]
    return key, default if limit_ppm is None else float(limit_ppm)


def find_min_ventilation(
    occupancy: np.ndarray,
    params: RoomParameters,
    dt_s: float,
    criterion: str = "8h",
    limit_ppm: float | None = None,
    bracket_lps: tuple[float, float] = (1.0, 1000.0),
    method: str = "secant",
    x_tol: float = 0.01,
    f_tol: float = 0.01,
    variable: str = "direct",
) -> RootResult:
    """Smallest constant outdoor-air flow ``Q`` [L/s] that meets a CO2 criterion.

    Solves ``F(Q) = metric(Q) - limit = 0`` (see the module docstring for why
    ``F`` is non-increasing in ``Q``).

    Parameters
    ----------
    occupancy : array_like
        People per step over the assessment period (at least 8 hours).
    params : RoomParameters
        Model parameters.
    dt_s : float
        Step length [s].
    criterion : {"8h", "peak"}, default "8h"
        ``"8h"``: worst 8-hour mean CO2 (NCC limit 850 ppm by default).
        ``"peak"``: worst step-mean CO2 (1000 ppm by default).
    limit_ppm : float, optional
        Overrides the criterion's default limit.
    bracket_lps : (float, float), default (1, 1000)
        Starting bracket. The upper end is doubled automatically until the
        criterion is met.
    method : {"secant", "bisection"}, default "secant"
        Root finder.
    x_tol : float, default 0.01
        Tolerance on ``Q`` [L/s].
    f_tol : float, default 0.01
        Tolerance on the metric [ppm].
    variable : {"direct", "inverse"}, default "direct"
        Root-find in ``Q`` or in ``1/Q``. ``"inverse"`` makes ``F`` nearly
        linear, so the secant method converges in fewer ODE solves.

    Returns
    -------
    RootResult
        ``root`` is ``Q_min`` [L/s]. ``history`` holds every ``(Q, F)`` tried.

    Raises
    ------
    ValueError
        For an unknown criterion or method, or if the criterion is already
        met at the lower bracket end.
    """
    key, limit = _criterion(criterion, limit_ppm)
    occ, _ = validate_inputs(occupancy, 1.0, dt_s)

    def F(q: float) -> float:
        return evaluate_ventilation(occ, q, params, dt_s)[key] - limit

    return _solve_monotone_decreasing(F, bracket_lps[0], bracket_lps[1], method, x_tol, f_tol,
                                      variable=variable)


# ---------------------------------------------------------------------------
# Demand-controlled ventilation
# ---------------------------------------------------------------------------

def dcv_ventilation(
    occupancy: np.ndarray,
    q_per_person_lps: float,
    q_floor_lps: float,
    q_cap_lps: float | None = None,
) -> np.ndarray:
    """Demand-controlled supply ``Q(t) = min(Q_cap, max(Q_floor, q_p * N(t)))`` [L/s].

    This is an idealised controller that responds instantly to the measured
    count. The floor keeps a minimum flow when the room is empty or lightly
    used, so CO2 (and other pollutants) can still be flushed out. The optional
    cap represents the maximum capacity of the fan or air-handling unit.

    Raises
    ------
    ValueError
        If a rate is negative, or the cap is below the floor.
    """
    if q_per_person_lps < 0 or q_floor_lps < 0:
        raise ValueError("q_per_person_lps and q_floor_lps must be non-negative")
    if q_cap_lps is not None and q_cap_lps < q_floor_lps:
        raise ValueError("q_cap_lps must be at least q_floor_lps")
    occ = np.asarray(occupancy, dtype=np.float64)
    q = np.maximum(q_floor_lps, q_per_person_lps * occ)
    return q if q_cap_lps is None else np.minimum(q, q_cap_lps)


def find_min_dcv_rate(
    occupancy: np.ndarray,
    params: RoomParameters,
    dt_s: float,
    q_floor_lps: float,
    q_cap_lps: float | None = None,
    criterion: str = "8h",
    limit_ppm: float | None = None,
    bracket_lps_per_person: tuple[float, float] = (0.1, 50.0),
    method: str = "secant",
    x_tol: float = 1e-3,
    f_tol: float = 0.01,
) -> RootResult:
    """Smallest per-person DCV rate ``q_p`` [L/s per person] that meets a CO2 criterion.

    ``Q(t) = min(q_cap, max(q_floor, q_p N(t)))`` never decreases at any
    instant as ``q_p`` rises, so the metric is non-increasing in ``q_p`` and
    bracketing applies. Parameters and return value are as for
    :func:`find_min_ventilation`, with ``root`` in L/s per person. If a cap is
    set and the criterion cannot be met even at the bracket's upper end, a
    ``ValueError`` is raised.
    """
    key, limit = _criterion(criterion, limit_ppm)
    occ, _ = validate_inputs(occupancy, 1.0, dt_s)

    def F(qp: float) -> float:
        q = dcv_ventilation(occ, qp, q_floor_lps, q_cap_lps)
        return evaluate_ventilation(occ, q, params, dt_s)[key] - limit

    return _solve_monotone_decreasing(F, bracket_lps_per_person[0], bracket_lps_per_person[1],
                                      method, x_tol, f_tol)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def outdoor_air_energy(
    ventilation_lps: np.ndarray | float,
    dt_s: float,
    n_steps: int,
    delta_t_k: float,
    rho_cp_kj_per_m3k: float = RHO_CP_AIR_KJ_PER_M3K,
) -> dict[str, float]:
    """Outdoor-air volume and the sensible energy needed to condition it.

    ``E = rho c_p * integral(Q dt) * dT``. Because ``Q`` is constant within each
    step, the integral is the exact sum ``sum(Q_i dt)``: the rectangle rule is
    exact for a piecewise-constant integrand (Week 6).

    Parameters
    ----------
    ventilation_lps : float or array_like
        Flow [L/s], constant or one value per step.
    dt_s : float
        Step length [s].
    n_steps : int
        Number of steps (used when ``ventilation_lps`` is a constant).
    delta_t_k : float
        Representative magnitude of the indoor–outdoor temperature difference
        [K] (an assumption; no outdoor temperature data are available).
    rho_cp_kj_per_m3k : float, default 1.206
        Volumetric heat capacity of air [kJ/(m3 K)].

    Returns
    -------
    dict
        ``air_volume_m3`` and ``energy_kwh``.

    Raises
    ------
    ValueError
        For negative flows or temperature difference, non-positive ``dt_s``,
        or a length mismatch.
    """
    q = np.asarray(ventilation_lps, dtype=np.float64)
    if q.ndim == 0:
        q = np.full(int(n_steps), float(q))
    if q.size != n_steps:
        raise ValueError(f"ventilation has {q.size} values but n_steps={n_steps}")
    if np.any(q < 0) or delta_t_k < 0 or dt_s <= 0:
        raise ValueError("flows and delta_t_k must be non-negative and dt_s positive")
    volume = float(np.sum(q) / 1000.0 * dt_s)
    energy_kwh = volume * rho_cp_kj_per_m3k * delta_t_k / 3600.0
    return {"air_volume_m3": volume, "energy_kwh": energy_kwh}
