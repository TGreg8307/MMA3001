"""Well-mixed single-zone CO2 mass-balance model and its exact solution.

Governing equation
------------------
For a room of air volume ``V`` [m3] that is ventilated with outdoor air at
``Q`` [m3/s] and occupied by ``N`` people, each exhaling CO2 at ``G``
[m3/s per person], conservation of CO2 gives::

    V dC/dt = 1e6 * G * N(t) + Q(t) * (C_out - C)

with ``C`` the indoor concentration in ppm (the 1e6 converts a volume
fraction to ppm). Dividing by ``V`` puts it in the linear form::

    dC/dt = a(t) - k(t) * C,
    a = (1e6 G N + Q C_out) / V   [ppm/s],
    k = Q / V                     [1/s]

where ``k`` is the air-change rate and ``tau = 1/k`` the time constant.

Exact solution
--------------
``N`` (from the resampled sensor data) and ``Q`` are constant within each
time step of length ``h``, so ``a`` and ``k`` are constant on each step and
the ODE has the exact step solution::

    C_{i+1} = E_i * C_i + a_i * h * phi1(k_i h),   E_i = exp(-k_i h)

with ``phi1(x) = (1 - exp(-x)) / x`` (``phi1(0) = 1``, so ``Q = 0`` is
handled smoothly). Applying it step by step gives the solution at every grid
point with **no discretisation error**. Only floating-point round-off
remains. It is the reference against which the numerical solvers in
:mod:`occvent.solvers` are verified.

Units used in the public interface: volumes in m3, flow rates in **L/s**
(the unit used by AS 1668.2), concentrations in ppm and times in seconds.

Assumptions (also stated in the report):

* the room air is perfectly mixed;
* occupants are the only CO2 source;
* the outdoor concentration is constant;
* sensor counts are accurate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import DEFAULT_CO2_GENERATION_LPS, DEFAULT_OUTDOOR_CO2_PPM

#: Conversion factor from a volume fraction to parts per million.
PPM: float = 1.0e6
#: Litres per cubic metre.
L_PER_M3: float = 1000.0


@dataclass(frozen=True)
class RoomParameters:
    """Physical parameters of the single-zone CO2 model.

    Attributes
    ----------
    volume_m3 : float
        Internal air volume ``V`` [m3]. Must be > 0.
    outdoor_co2_ppm : float, default 420
        CO2 concentration of the supply (outdoor) air ``C_out`` [ppm]. Must be >= 0.
    co2_generation_lps : float, default 0.0052
        CO2 exhaled per occupant ``G`` [L/s per person]. Must be > 0.
    """

    volume_m3: float
    outdoor_co2_ppm: float = DEFAULT_OUTDOOR_CO2_PPM
    co2_generation_lps: float = DEFAULT_CO2_GENERATION_LPS

    def __post_init__(self) -> None:
        if not np.isfinite(self.volume_m3) or self.volume_m3 <= 0:
            raise ValueError(f"volume_m3 must be a positive number, got {self.volume_m3}")
        if not np.isfinite(self.outdoor_co2_ppm) or self.outdoor_co2_ppm < 0:
            raise ValueError(f"outdoor_co2_ppm must be >= 0, got {self.outdoor_co2_ppm}")
        if not np.isfinite(self.co2_generation_lps) or self.co2_generation_lps <= 0:
            raise ValueError(f"co2_generation_lps must be > 0, got {self.co2_generation_lps}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def phi1(x: np.ndarray | float) -> np.ndarray:
    """Evaluate ``phi1(x) = (1 - exp(-x)) / x`` accurately, with ``phi1(0) = 1``.

    ``-expm1(-x)`` avoids the cancellation in ``1 - exp(-x)`` for small ``x``.

    Parameters
    ----------
    x : array_like
        Non-negative arguments (``k * h``).

    Returns
    -------
    numpy.ndarray
        ``phi1(x)`` element-wise, in (0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    nz = x != 0.0
    out[nz] = -np.expm1(-x[nz]) / x[nz]
    return out


def phi2(x: np.ndarray | float) -> np.ndarray:
    """Evaluate ``phi2(x) = (x - 1 + exp(-x)) / x**2`` accurately, with ``phi2(0) = 1/2``.

    Used for the exact time-average of ``C`` over a step. For ``x < 1e-3``,
    a 4-term Taylor series replaces the direct formula, which loses accuracy
    to cancellation there. The truncation error is below ``x**4 / 720 < 2e-15``.

    Parameters
    ----------
    x : array_like
        Non-negative arguments (``k * h``).

    Returns
    -------
    numpy.ndarray
        ``phi2(x)`` element-wise, in (0, 1/2].
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = x < 1e-3
    xs = x[small]
    out[small] = 0.5 - xs / 6.0 + xs**2 / 24.0 - xs**3 / 120.0
    xl = x[~small]
    out[~small] = (xl + np.expm1(-xl)) / xl**2
    return out


def validate_inputs(
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Check model inputs and broadcast ventilation to one value per step.

    Parameters
    ----------
    occupancy : array_like
        People in the room during each step, ``N_i >= 0`` (may be
        fractional, since resampled bins are time-averages). Length ``n >= 1``.
    ventilation_lps : float or array_like
        Outdoor-air flow ``Q`` [L/s] >= 0: one constant value, or one per step.
    dt_s : float
        Step length [s] > 0.

    Returns
    -------
    occupancy, ventilation : numpy.ndarray
        Float64 arrays of length ``n``.

    Raises
    ------
    ValueError
        If any value is negative or non-finite, the lengths do not match,
        the occupancy is empty, or ``dt_s`` is not positive.
    """
    n_arr = np.asarray(occupancy, dtype=np.float64)
    if n_arr.ndim != 1 or n_arr.size == 0:
        raise ValueError("occupancy must be a non-empty 1-D array")
    if not np.all(np.isfinite(n_arr)) or np.any(n_arr < 0):
        raise ValueError("occupancy must be finite and non-negative (fill or drop NaN bins first)")
    q_arr = np.asarray(ventilation_lps, dtype=np.float64)
    if q_arr.ndim == 0:
        q_arr = np.full(n_arr.size, float(q_arr))
    if q_arr.shape != n_arr.shape:
        raise ValueError(f"ventilation has {q_arr.size} values but occupancy has {n_arr.size}")
    if not np.all(np.isfinite(q_arr)) or np.any(q_arr < 0):
        raise ValueError("ventilation_lps must be finite and non-negative")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    return n_arr, q_arr


def ode_coefficients(
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-step coefficients ``a`` [ppm/s] and ``k`` [1/s] of ``dC/dt = a - k C``.

    Parameters
    ----------
    occupancy, ventilation_lps, dt_s
        See :func:`validate_inputs`.
    params : RoomParameters
        Room volume, outdoor CO2 and generation rate.

    Returns
    -------
    a, k : numpy.ndarray
        Arrays of length ``n``.
    """
    n_arr, q_arr = validate_inputs(occupancy, ventilation_lps, dt_s)
    q_m3s = q_arr / L_PER_M3
    g_m3s = params.co2_generation_lps / L_PER_M3
    a = (PPM * g_m3s * n_arr + q_m3s * params.outdoor_co2_ppm) / params.volume_m3
    k = q_m3s / params.volume_m3
    return a, k


def steady_state_ppm(occupancy: float, ventilation_lps: float, params: RoomParameters) -> float:
    """Steady-state concentration ``C_ss = C_out + 1e6 G N / Q`` for constant ``N`` and ``Q``.

    ``V`` cancels, so the steady state depends only on the ventilation rate
    per person. For example, 10 L/s per person with G = 0.0052 L/s gives
    C_ss = C_out + 520 ppm.

    Parameters
    ----------
    occupancy : float
        People, >= 0.
    ventilation_lps : float
        Outdoor-air flow [L/s], > 0.
    params : RoomParameters
        Model parameters.

    Returns
    -------
    float
        ``C_ss`` [ppm].

    Raises
    ------
    ValueError
        If ``ventilation_lps <= 0`` (no steady state exists) or ``occupancy < 0``.
    """
    if ventilation_lps <= 0:
        raise ValueError("a steady state exists only for ventilation_lps > 0")
    if occupancy < 0:
        raise ValueError("occupancy must be non-negative")
    return params.outdoor_co2_ppm + PPM * params.co2_generation_lps * occupancy / ventilation_lps


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def exact_solution(
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
    c0_ppm: float | None = None,
    method: str = "auto",
) -> np.ndarray:
    """Exact CO2 concentration at every grid point, for piecewise-constant N and Q.

    Evaluates the linear recurrence ``C_{i+1} = E_i C_i + B_i`` with
    ``E_i = exp(-k_i h)`` and ``B_i = a_i h phi1(k_i h)``.

    Two equivalent evaluations are available:

    * ``"loop"``: a plain sequential loop. Works for any ``Q(t)``.
    * ``"lfilter"``: when ``Q`` (and so ``E``) is constant, the recurrence
      is a first-order IIR filter, evaluated in compiled code by
      :func:`scipy.signal.lfilter`. This is much faster for long series.

    ``"auto"`` uses ``"lfilter"`` when ``Q`` is constant, otherwise ``"loop"``.

    Parameters
    ----------
    occupancy : array_like
        People during each step (length ``n``).
    ventilation_lps : float or array_like
        Outdoor-air flow [L/s], constant or one value per step.
    params : RoomParameters
        Model parameters.
    dt_s : float
        Step length [s].
    c0_ppm : float, optional
        Concentration at the first grid point. Defaults to the outdoor
        concentration (a room that has been empty for a long time).
    method : {"auto", "loop", "lfilter"}
        Evaluation method (see above).

    Returns
    -------
    numpy.ndarray
        ``C`` [ppm] at the ``n + 1`` grid points: the start of each step,
        plus the end of the last step.

    Raises
    ------
    ValueError
        For invalid inputs, an unknown ``method``, or ``"lfilter"`` requested
        with a time-varying ``Q``.
    """
    a, k = ode_coefficients(occupancy, ventilation_lps, params, dt_s)
    c0 = params.outdoor_co2_ppm if c0_ppm is None else float(c0_ppm)
    if not np.isfinite(c0) or c0 < 0:
        raise ValueError(f"c0_ppm must be finite and >= 0, got {c0_ppm}")
    x = k * dt_s
    e = np.exp(-x)
    b = a * dt_s * phi1(x)
    constant_q = bool(np.all(k == k[0]))

    if method == "auto":
        method = "lfilter" if constant_q else "loop"
    if method == "lfilter":
        if not constant_q:
            raise ValueError("method='lfilter' requires a constant ventilation rate")
        e0 = float(e[0])
        # y[i] = b[i] + e0 * y[i-1], with y[-1] = c0  ->  initial filter state e0 * c0.
        y, _ = lfilter([1.0], [1.0, -e0], b, zi=[e0 * c0])
        return np.concatenate(([c0], y))
    if method == "loop":
        out = np.empty(a.size + 1)
        out[0] = c = c0
        for i, (ei, bi) in enumerate(zip(e.tolist(), b.tolist()), start=1):
            c = ei * c + bi
            out[i] = c
        return out
    raise ValueError(f"unknown method {method!r}; use 'auto', 'loop' or 'lfilter'")


def step_mean_concentration(
    c_ppm: np.ndarray,
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
) -> np.ndarray:
    """Exact time-average of ``C`` over each step, given the exact grid values.

    Over a step that starts at ``C_i``::

        mean_i = C_i * phi1(k h) + a h * phi2(k h)

    This is the quantity that enters averaged criteria such as the
    NCC 8-hour mean. Averaging the grid-point values instead would add a
    small bias.

    Parameters
    ----------
    c_ppm : numpy.ndarray
        Grid values from :func:`exact_solution` (length ``n + 1``).
    occupancy, ventilation_lps, params, dt_s
        As passed to :func:`exact_solution`.

    Returns
    -------
    numpy.ndarray
        Mean concentration in each step [ppm], length ``n``.

    Raises
    ------
    ValueError
        If the inputs are invalid (see :func:`validate_inputs`) or ``c_ppm``
        does not have exactly one more value than ``occupancy``.
    """
    a, k = ode_coefficients(occupancy, ventilation_lps, params, dt_s)
    c_ppm = np.asarray(c_ppm, dtype=np.float64)
    if c_ppm.size != a.size + 1:
        raise ValueError("c_ppm must have one more value than occupancy")
    x = k * dt_s
    return c_ppm[:-1] * phi1(x) + a * dt_s * phi2(x)


def co2_mass_balance(
    c_ppm: np.ndarray,
    occupancy: np.ndarray,
    ventilation_lps: np.ndarray | float,
    params: RoomParameters,
    dt_s: float,
) -> dict[str, float]:
    """Global CO2 budget of a simulation: generated = removed + accumulated.

    All terms are volumes of pure CO2 [m3]:

    * generated   = sum(G N_i h)
    * removed     = sum(Q_i (mean_i - C_out) h) / 1e6
      (net CO2 carried out by ventilation)
    * accumulated = V (C_end - C_start) / 1e6

    For the exact solution the residual is zero up to round-off, because
    the step solution integrates the ODE exactly. This is a conservation
    check on the implementation.

    Parameters
    ----------
    c_ppm : numpy.ndarray
        Concentration at the ``n + 1`` grid points [ppm], from any solver.
    occupancy, ventilation_lps, params, dt_s
        As passed to the solver that produced ``c_ppm``.

    Returns
    -------
    dict
        ``generated_m3``, ``removed_m3``, ``accumulated_m3``,
        ``residual_m3`` (generated - removed - accumulated) and
        ``relative_residual`` (residual / generated, or 0 if nothing was generated).
    """
    n_arr, q_arr = validate_inputs(occupancy, ventilation_lps, dt_s)
    mean = step_mean_concentration(c_ppm, n_arr, q_arr, params, dt_s)
    generated = float(np.sum(params.co2_generation_lps / L_PER_M3 * n_arr * dt_s))
    removed = float(np.sum(q_arr / L_PER_M3 * (mean - params.outdoor_co2_ppm) * dt_s) / PPM)
    accumulated = float(params.volume_m3 * (c_ppm[-1] - c_ppm[0]) / PPM)
    residual = generated - removed - accumulated
    return {
        "generated_m3": generated,
        "removed_m3": removed,
        "accumulated_m3": accumulated,
        "residual_m3": residual,
        "relative_residual": residual / generated if generated > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# From a resampled occupancy series to model input
# ---------------------------------------------------------------------------

def occupancy_from_grid(grid: pd.DataFrame, fill: str = "error") -> tuple[np.ndarray, float, int]:
    """Extract the model input ``N`` and step ``dt`` from a resampled series.

    Parameters
    ----------
    grid : pandas.DataFrame
        Output of :func:`occvent.cleaning.resample_occupancy` (or
        :func:`occvent.io.load_processed`). It must have a regular,
        tz-aware ``DatetimeIndex`` and a ``headcount`` column.
    fill : {"error", "zero", "ffill"}
        How to handle bins with no observation (``headcount`` is NaN):

        * ``"error"`` (default): raise, so gaps are never filled silently.
        * ``"zero"``: assume the room was empty. This under-estimates CO2.
        * ``"ffill"``: carry the last observed value forward.

    Returns
    -------
    occupancy : numpy.ndarray
        People per step (float64, no NaN).
    dt_s : float
        Step length [s].
    n_filled : int
        Number of bins that were filled.

    Raises
    ------
    ValueError
        If the index is irregular or has fewer than two points, ``fill`` is
        unknown, ``fill="error"`` and gaps exist, or ``fill="ffill"`` cannot
        fill leading gaps.
    """
    if not isinstance(grid.index, pd.DatetimeIndex) or len(grid.index) < 2:
        raise ValueError("grid needs a DatetimeIndex with at least two rows")
    steps = grid.index[1:] - grid.index[:-1]  # TimedeltaIndex
    if not (steps == steps[0]).all() or steps[0] <= pd.Timedelta(0):
        raise ValueError("grid index must be regularly spaced and increasing")
    dt_s = steps[0].total_seconds()

    hc = grid["headcount"].astype("float64")
    missing = hc.isna()
    n_filled = int(missing.sum())
    if n_filled:
        if fill == "error":
            raise ValueError(f"{n_filled} bins have no occupancy; choose fill='zero' or 'ffill'")
        if fill == "zero":
            hc = hc.fillna(0.0)
        elif fill == "ffill":
            hc = hc.ffill()
            if hc.isna().any():
                raise ValueError("fill='ffill' cannot fill gaps at the start of the series")
        else:
            raise ValueError(f"unknown fill {fill!r}; use 'error', 'zero' or 'ffill'")
    elif fill not in ("error", "zero", "ffill"):
        raise ValueError(f"unknown fill {fill!r}; use 'error', 'zero' or 'ffill'")
    return hc.to_numpy(), float(dt_s), n_filled
