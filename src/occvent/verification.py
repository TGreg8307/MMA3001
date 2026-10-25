"""Tools for verifying the solvers and analysing their performance.

They support Stage 3 of the project (``notebooks/03_verification.ipynb``),
following MMA3001 Weeks 2, 7 and 9:

* **Convergence:** error against the exact solution as the step ``h`` is
  refined, and the observed order ``p_obs = log(E1/E2) / log(h1/h2)``
  (Notebook 7.3.7).
* **Stability:** the stability (amplification) function ``R`` of each method
  for the decay test equation ``y' = -lambda y``, with ``r = lambda h``
  (Notebooks 7.3.8, 7.4.8, 9.1, 9.3).
* **Stiffness:** the eigenvalue and time constant of the CO2 model, and the
  step limits they impose (Notebook 9.2).
* **Cost:** floating-point operations (FLOPs), right-hand-side evaluations,
  bytes moved per step, arithmetic intensity, timing, and an empirical
  roofline of the machine (Notebook 2.1).
"""

from __future__ import annotations

import time
import timeit
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .co2model import RoomParameters, exact_solution
from .solvers import SOLVERS

# ---------------------------------------------------------------------------
# Operation counts (derived by hand from the update formulas in solvers.py)
# ---------------------------------------------------------------------------

#: Floating-point operations per solver sub-step, counted from the code:
#:
#: * explicit Euler ``c + h*(a - k*c)``: 1 mul, 1 sub, 1 mul, 1 add = 4
#: * implicit Euler ``(c + ha) / denom`` (``ha``, ``denom`` precomputed per data step): 1 add, 1 div = 2
#: * RK4: ``k1`` 2, ``k2`` 4, ``k3`` 4, ``k4`` 4, update 7 = 21
#: * exact recurrence ``E*c + B`` (``E``, ``B`` precomputed, vectorised): 2
FLOPS_PER_SUBSTEP: dict[str, int] = {
    "explicit_euler": 4,
    "implicit_euler": 2,
    "rk4": 21,
    "exact": 2,
}

#: Right-hand-side evaluations per sub-step. Implicit Euler's closed-form
#: linear solve is counted as one evaluation.
RHS_EVALS_PER_SUBSTEP: dict[str, int] = {
    "explicit_euler": 1,
    "implicit_euler": 1,
    "rk4": 4,
}

#: Main-memory traffic per *data* step [bytes]: load two float64 coefficients
#: (``a_i``, ``k_i``, or ``E_i``, ``B_i``) and store one float64 result. The
#: running state ``C`` stays in a register, as in the SAXPY example of
#: Notebook 2.1.6.
BYTES_PER_STEP: int = 3 * 8

#: Largest dimensionless step ``r = k h`` on the negative real axis for which
#: each method's stability function satisfies ``|R| <= 1``. The RK4 value is
#: the positive root of ``r^3/24 - r^2/6 + r/2 - 1 = 0``.
STABILITY_LIMIT: dict[str, float] = {
    "explicit_euler": 2.0,
    "rk4": 2.7852935634052813,
    "implicit_euler": float("inf"),
    "exact": float("inf"),
}


def stability_function(method: str, r: np.ndarray | float) -> np.ndarray:
    """Amplification factor ``R(-r)`` for the decay test equation ``y' = -lambda y``.

    One step of size ``h`` multiplies the solution (and any error) by
    ``R(-r)``, with ``r = lambda h >= 0``:

    * explicit Euler: ``1 - r``
    * implicit Euler: ``1 / (1 + r)``
    * RK4: ``1 - r + r^2/2 - r^3/6 + r^4/24``
    * exact: ``exp(-r)``

    The method is stable when ``|R| <= 1``. It is free of artificial
    oscillation only if additionally ``R >= 0``.

    Parameters
    ----------
    method : {"explicit_euler", "implicit_euler", "rk4", "exact"}
        Method name.
    r : array_like
        Dimensionless step(s) ``r = k h``, >= 0.

    Returns
    -------
    numpy.ndarray
        ``R(-r)``, element-wise.

    Raises
    ------
    ValueError
        If ``method`` is unknown or any ``r`` is negative.
    """
    r = np.asarray(r, dtype=np.float64)
    if np.any(r < 0):
        raise ValueError("r = k*h must be non-negative")
    if method == "explicit_euler":
        return 1.0 - r
    if method == "implicit_euler":
        return 1.0 / (1.0 + r)
    if method == "rk4":
        return 1.0 - r + r**2 / 2.0 - r**3 / 6.0 + r**4 / 24.0
    if method == "exact":
        return np.exp(-r)
    raise ValueError(f"unknown method {method!r}")


def observed_orders(h: Sequence[float], errors: Sequence[float]) -> np.ndarray:
    """Observed order of convergence between successive refinements.

    ``p_obs = log(E_i / E_{i+1}) / log(h_i / h_{i+1})`` (Notebook 7.3.7).

    Parameters
    ----------
    h : sequence of float
        Step sizes, all positive and distinct.
    errors : sequence of float
        Corresponding errors, all positive.

    Returns
    -------
    numpy.ndarray
        ``len(h) - 1`` observed orders.

    Raises
    ------
    ValueError
        If the lengths differ, fewer than two points are given, or any value
        is not positive.
    """
    h_arr = np.asarray(h, dtype=np.float64)
    e_arr = np.asarray(errors, dtype=np.float64)
    if h_arr.shape != e_arr.shape or h_arr.size < 2:
        raise ValueError("h and errors must have the same length (at least 2)")
    if np.any(h_arr <= 0) or np.any(e_arr <= 0):
        raise ValueError("step sizes and errors must be positive")
    return np.log(e_arr[:-1] / e_arr[1:]) / np.log(h_arr[:-1] / h_arr[1:])


def convergence_study(
    method: str,
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
    substeps: Sequence[int],
    c0_ppm: float | None = None,
) -> pd.DataFrame:
    """Error, cost and observed order of a fixed-step solver as ``h`` is refined.

    The reference is :func:`occvent.co2model.exact_solution` on the same data
    grid, so the error measured is purely the solver's discretisation (plus
    round-off) error.

    Parameters
    ----------
    method : {"explicit_euler", "implicit_euler", "rk4"}
        Fixed-step solver from :data:`occvent.solvers.SOLVERS`.
    occupancy, ventilation_lps, params, dt_s, c0_ppm
        Model inputs (see :func:`occvent.co2model.exact_solution`).
    substeps : sequence of int
        Sub-steps per data step to test (``h = dt_s / m``), in increasing order.

    Returns
    -------
    pandas.DataFrame
        One row per ``m``, with columns ``substeps``, ``h_s``, ``kh`` (largest
        ``k h``), ``max_abs_error_ppm``, ``rms_error_ppm``, ``rhs_evaluations``,
        ``flops``, ``runtime_s`` and ``observed_order`` (NaN on the first row).

    Raises
    ------
    ValueError
        If ``method`` is not a fixed-step solver, or ``substeps`` is not
        strictly increasing.
    """
    if method not in RHS_EVALS_PER_SUBSTEP:
        raise ValueError(f"method must be one of {sorted(RHS_EVALS_PER_SUBSTEP)}, got {method!r}")
    m_list = [int(m) for m in substeps]
    if any(b <= a for a, b in zip(m_list, m_list[1:])) or m_list[0] < 1:
        raise ValueError("substeps must be strictly increasing positive integers")
    solver = SOLVERS[method]
    ref = exact_solution(occupancy, ventilation_lps, params, dt_s, c0_ppm)
    n_steps = ref.size - 1
    k_max = float(np.max(np.asarray(ventilation_lps, dtype=float))) / 1000.0 / params.volume_m3

    rows = []
    for m in m_list:
        t0 = time.perf_counter()
        c = solver(occupancy, ventilation_lps, params, dt_s, substeps=m, c0_ppm=c0_ppm)
        runtime = time.perf_counter() - t0
        err = c - ref
        rows.append({
            "substeps": m,
            "h_s": dt_s / m,
            "kh": k_max * dt_s / m,
            "max_abs_error_ppm": float(np.max(np.abs(err))),
            "rms_error_ppm": float(np.sqrt(np.mean(err**2))),
            "rhs_evaluations": n_steps * m * RHS_EVALS_PER_SUBSTEP[method],
            "flops": n_steps * m * FLOPS_PER_SUBSTEP[method],
            "runtime_s": runtime,
        })
    df = pd.DataFrame(rows)
    orders = np.full(len(df), np.nan)
    errs = df["max_abs_error_ppm"].to_numpy()
    if np.all(errs > 0):
        orders[1:] = observed_orders(df["h_s"].to_numpy(), errs)
    df["observed_order"] = orders
    return df


def arithmetic_intensity(method: str, substeps: int = 1) -> float:
    """Arithmetic intensity [FLOP/byte] of one data step of a solver.

    ``AI = FLOPS_PER_SUBSTEP[method] * substeps / BYTES_PER_STEP``. Sub-steps
    reuse the same loaded coefficients, so refining ``h`` raises the
    intensity.

    Parameters
    ----------
    method : str
        A key of :data:`FLOPS_PER_SUBSTEP` (``"explicit_euler"``,
        ``"implicit_euler"``, ``"rk4"`` or ``"exact"``).
    substeps : int, default 1
        Solver sub-steps per data step.

    Returns
    -------
    float
        FLOPs per byte of memory traffic.

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``substeps < 1``.
    """
    if method not in FLOPS_PER_SUBSTEP:
        raise ValueError(f"unknown method {method!r}")
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    return FLOPS_PER_SUBSTEP[method] * substeps / BYTES_PER_STEP


def best_time(fn: Callable[[], object], repeat: int = 5, number: int = 1) -> float:
    """Best (minimum) wall-clock time of ``fn()`` in seconds, using :mod:`timeit`.

    This is the same measurement as the ``%timeit`` magic used in the unit.
    The minimum is reported because it is least affected by other processes.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to time (wrap arguments with ``lambda``).
    repeat : int, default 5
        Number of independent timing runs.
    number : int, default 1
        Calls per run. The result is divided by ``number``.

    Returns
    -------
    float
        Seconds per call.

    Raises
    ------
    ValueError
        If ``repeat`` or ``number`` is less than 1.
    """
    if repeat < 1 or number < 1:
        raise ValueError("repeat and number must be >= 1")
    return min(timeit.repeat(fn, repeat=repeat, number=number)) / number


def empirical_roofline(matrix_n: int = 1024, copy_n: int = 2**23, repeat: int = 3) -> dict[str, float]:
    """Measure this machine's practical compute peak and memory bandwidth.

    Follows the empirical testing approach of Notebook 2.1.7:

    * **Peak FLOP/s** from a large NumPy matrix multiply (compute-bound,
      uses optimised multi-threaded BLAS): ``2 n^3`` FLOPs.
    * **Memory bandwidth** from copying a large float64 array (bandwidth-bound):
      ``2 * 8 * N`` bytes (one read and one write per element).

    Parameters
    ----------
    matrix_n : int, default 1024
        Matrix size for the FLOP test.
    copy_n : int, default 2**23
        Elements copied for the bandwidth test (2**23 float64 = 64 MB, larger
        than CPU caches).
    repeat : int, default 3
        Repetitions. The best time is used.

    Returns
    -------
    dict
        ``peak_flops`` [FLOP/s], ``bandwidth_bytes_per_s`` [B/s] and
        ``machine_balance`` [FLOP/byte] (= peak / bandwidth).
    """
    rng = np.random.default_rng(0)
    a = rng.random((matrix_n, matrix_n))
    b = rng.random((matrix_n, matrix_n))
    a @ b  # warm-up (start-up cost, Notebook 2.1.6)
    t_mm = best_time(lambda: a @ b, repeat=repeat)
    src = rng.random(copy_n)
    dst = np.empty_like(src)
    np.copyto(dst, src)
    t_cp = best_time(lambda: np.copyto(dst, src), repeat=repeat)
    peak = 2.0 * matrix_n**3 / t_mm
    bw = 2.0 * 8.0 * copy_n / t_cp
    return {"peak_flops": peak, "bandwidth_bytes_per_s": bw, "machine_balance": peak / bw}


def stiffness_summary(ventilation_lps: float, volume_m3: float, dt_s: float = 60.0) -> dict[str, float]:
    """Eigenvalue, time scale and step limits of the single-zone CO2 model.

    The homogeneous equation ``dC/dt = -k C`` has the single eigenvalue
    ``lambda = -k = -Q/V``. A problem is *stiff* when its time scales differ
    widely (Notebook 9.2). With one eigenvalue the stiffness ratio is 1, so
    the model is **not stiff**. Explicit methods are limited only by
    ``h < 2/k`` (Euler) or ``h < 2.785/k`` (RK4).

    Parameters
    ----------
    ventilation_lps : float
        Outdoor-air flow [L/s], > 0.
    volume_m3 : float
        Room volume [m3], > 0.
    dt_s : float, default 60
        Data step [s], used to report ``k * dt``.

    Returns
    -------
    dict
        ``eigenvalue_per_s``, ``time_constant_s``, ``air_changes_per_hour``,
        ``k_dt``, ``euler_max_stable_step_s``, ``rk4_max_stable_step_s`` and
        ``stiffness_ratio``.

    Raises
    ------
    ValueError
        If any argument is not positive.
    """
    if ventilation_lps <= 0 or volume_m3 <= 0 or dt_s <= 0:
        raise ValueError("ventilation_lps, volume_m3 and dt_s must be positive")
    k = ventilation_lps / 1000.0 / volume_m3
    return {
        "eigenvalue_per_s": -k,
        "time_constant_s": 1.0 / k,
        "air_changes_per_hour": k * 3600.0,
        "k_dt": k * dt_s,
        "euler_max_stable_step_s": STABILITY_LIMIT["explicit_euler"] / k,
        "rk4_max_stable_step_s": STABILITY_LIMIT["rk4"] / k,
        "stiffness_ratio": 1.0,
    }
