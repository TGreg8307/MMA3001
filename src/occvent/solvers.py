"""Numerical solvers for the CO2 model, verified against the exact solution.

All solvers integrate ``dC/dt = a_i - k_i C`` (see :mod:`occvent.co2model`),
where ``a_i`` and ``k_i`` are constant within data step ``i`` of length
``dt``. Each data step is split into ``m = substeps`` equal sub-steps of
size ``h = dt / m``. Sub-steps never straddle a change in ``N`` or ``Q``,
so the forcing is smooth within every sub-step and each method achieves its
theoretical order of accuracy.

Methods implemented from scratch:

``explicit_euler``  ``C <- C + h f(C)``
    First order. Applying it to the test equation ``C' = -k C`` multiplies
    the error by ``(1 - k h)`` each step, so it is **stable only for
    k h < 2** and oscillates for ``1 < k h < 2``.

``implicit_euler``  ``C <- (C + h a) / (1 + h k)``
    First order. Solved in closed form because the ODE is linear. The
    amplification factor ``1 / (1 + k h)`` is below 1 for every ``h``, so
    it is **unconditionally stable** (A-stable and L-stable) and never
    overshoots.

``rk4``  classical 4-stage Runge-Kutta
    Fourth order. Stable for ``k h`` up to about 2.785.

``solve_scipy``  :func:`scipy.integrate.solve_ivp`
    An established adaptive solver (RK45 by default, or LSODA, etc.), used
    as an independent benchmark.

Every solver returns ``C`` at the ``n + 1`` data-grid points, matching
:func:`occvent.co2model.exact_solution`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .co2model import RoomParameters, ode_coefficients


def _setup(occupancy, ventilation_lps, params, dt_s, substeps, c0_ppm):
    """Validate inputs and return ``(a, k, h, c0)`` with ``a`` and ``k`` as Python lists."""
    if not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError(f"substeps must be a positive integer, got {substeps!r}")
    a, k = ode_coefficients(occupancy, ventilation_lps, params, dt_s)
    c0 = params.outdoor_co2_ppm if c0_ppm is None else float(c0_ppm)
    if not np.isfinite(c0) or c0 < 0:
        raise ValueError(f"c0_ppm must be finite and >= 0, got {c0_ppm}")
    # Python floats are much faster than NumPy scalars inside scalar loops.
    return a.tolist(), k.tolist(), dt_s / int(substeps), c0


def explicit_euler(
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
    substeps: int = 1,
    c0_ppm: float | None = None,
) -> np.ndarray:
    """Integrate the CO2 model with the explicit (forward) Euler method.

    Parameters
    ----------
    occupancy : array_like
        People during each data step (length ``n``).
    ventilation_lps : float or array_like
        Outdoor-air flow [L/s], constant or one value per step.
    params : RoomParameters
        Model parameters.
    dt_s : float
        Data step [s].
    substeps : int, default 1
        Sub-steps per data step (solver step ``h = dt_s / substeps``).
    c0_ppm : float, optional
        Initial concentration. Defaults to the outdoor concentration.

    Returns
    -------
    numpy.ndarray
        ``C`` [ppm] at the ``n + 1`` data-grid points. If ``k h > 2`` the
        result grows without bound (numerical instability).
    """
    a, k, h, c = _setup(occupancy, ventilation_lps, params, dt_s, substeps, c0_ppm)
    out = np.empty(len(a) + 1)
    out[0] = c
    for i, (ai, ki) in enumerate(zip(a, k), start=1):
        for _ in range(substeps):
            c = c + h * (ai - ki * c)
        out[i] = c
    return out


def implicit_euler(
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
    substeps: int = 1,
    c0_ppm: float | None = None,
) -> np.ndarray:
    """Integrate the CO2 model with the implicit (backward) Euler method.

    The implicit update ``C_new = C + h (a - k C_new)`` is linear in
    ``C_new``, so it is solved exactly as ``C_new = (C + h a) / (1 + h k)``.
    No iteration is needed. Parameters and return value as for
    :func:`explicit_euler`. The result is stable for every ``h``.
    """
    a, k, h, c = _setup(occupancy, ventilation_lps, params, dt_s, substeps, c0_ppm)
    out = np.empty(len(a) + 1)
    out[0] = c
    for i, (ai, ki) in enumerate(zip(a, k), start=1):
        denom = 1.0 + h * ki
        ha = h * ai
        for _ in range(substeps):
            c = (c + ha) / denom
        out[i] = c
    return out


def rk4(
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
    substeps: int = 1,
    c0_ppm: float | None = None,
) -> np.ndarray:
    """Integrate the CO2 model with the classical fourth-order Runge-Kutta method.

    With ``f(C) = a - k C`` on a sub-step::

        k1 = f(C)
        k2 = f(C + h/2 k1)
        k3 = f(C + h/2 k2)
        k4 = f(C + h k3)
        C <- C + h/6 (k1 + 2 k2 + 2 k3 + k4)

    Parameters and return value as for :func:`explicit_euler`.
    """
    a, k, h, c = _setup(occupancy, ventilation_lps, params, dt_s, substeps, c0_ppm)
    out = np.empty(len(a) + 1)
    out[0] = c
    half = 0.5 * h
    sixth = h / 6.0
    for i, (ai, ki) in enumerate(zip(a, k), start=1):
        for _ in range(substeps):
            k1 = ai - ki * c
            k2 = ai - ki * (c + half * k1)
            k3 = ai - ki * (c + half * k2)
            k4 = ai - ki * (c + h * k3)
            c = c + sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i] = c
    return out


def solve_scipy(
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
    substeps: int = 1,
    c0_ppm: float | None = None,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-6,
    full_output: bool = False,
):
    """Integrate the CO2 model with SciPy's adaptive :func:`~scipy.integrate.solve_ivp`.

    The right-hand side looks up the coefficients of the data step that
    contains ``t``. ``max_step = dt_s`` stops the adaptive controller from
    stepping over a whole occupancy pulse. The step-size control still has
    to "discover" each jump in ``N``, which is the main cost.

    Parameters
    ----------
    occupancy, ventilation_lps, params, dt_s, c0_ppm
        As for :func:`explicit_euler`.
    substeps : int, default 1
        Accepted for a common interface. Ignored, because step size is chosen
        adaptively from ``rtol`` and ``atol``.
    method : str, default "RK45"
        Any ``solve_ivp`` method, e.g. ``"RK45"``, ``"DOP853"``, ``"LSODA"``, ``"Radau"``.
    rtol, atol : float
        Relative and absolute error tolerances (``atol`` in ppm).
    full_output : bool, default False
        If True, also return a dict with the number of right-hand-side
        evaluations (``nfev``) and SciPy's status message.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, dict)
        ``C`` [ppm] at the ``n + 1`` data-grid points, plus the info dict
        if ``full_output``.

    Raises
    ------
    RuntimeError
        If the integration fails.
    """
    a, k, _, c0 = _setup(occupancy, ventilation_lps, params, dt_s, 1, c0_ppm)
    a_arr, k_arr = np.asarray(a), np.asarray(k)
    n = a_arr.size
    t_grid = np.arange(n + 1) * dt_s

    def rhs(t: float, c: np.ndarray) -> np.ndarray:
        i = min(int(t // dt_s), n - 1)
        return a_arr[i] - k_arr[i] * c

    sol = solve_ivp(rhs, (0.0, t_grid[-1]), [c0], method=method, t_eval=t_grid,
                    rtol=rtol, atol=atol, max_step=dt_s)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    c = sol.y[0].copy()
    if full_output:
        return c, {"nfev": int(sol.nfev), "message": sol.message, "method": method}
    return c


#: Registry of solvers with a common signature, used by verification code.
SOLVERS: dict[str, Callable[..., np.ndarray]] = {
    "explicit_euler": explicit_euler,
    "implicit_euler": implicit_euler,
    "rk4": rk4,
    "scipy_rk45": solve_scipy,
}

#: Theoretical order of accuracy of each fixed-step solver.
THEORETICAL_ORDER: dict[str, int] = {"explicit_euler": 1, "implicit_euler": 1, "rk4": 4}
