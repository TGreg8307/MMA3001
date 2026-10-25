"""Sensitivity and uncertainty analysis of the ventilation results.

The model inputs that cannot be measured from the data are uncertain:

* the room volume ``V`` (estimated);
* the CO2 generation rate per person ``G`` (literature range);
* the outdoor CO2 concentration ``C_out`` (assumed);
* a systematic people-counter error, a factor ``k`` on every count.

This module provides two taught approaches:

1. **One-at-a-time (OAT) perturbation** (:func:`oat_sensitivity`,
   :func:`elasticities`). Move one input while holding the others at their
   nominal values, and record the change in an output. The results are
   shown as a tornado chart.
2. **Monte Carlo uncertainty propagation** (:func:`sample_inputs`,
   :func:`percentile_summary`). Draw all inputs at random from assumed
   distributions, evaluate the output for every draw, and report the spread.

Exact scaling law (why Monte Carlo is cheap here)
-------------------------------------------------
The CO2 model is linear. Write ``C = C_out + Y`` with ``Y(0) = 0`` (the room
starts at outdoor level). With the counts scaled by ``k``::

    V dY/dt = 1e6 G k N(t) - Q Y.

Let ``Z`` solve the same equation for a *unit room* (``V = 1``, ``G = 1``,
``k = 1``, ``C_out = 0``) with flow ``a = Q / V``::

    dZ/dt = 1e6 N(t) - a Z.

Substituting ``Y = (k G / V) Z`` satisfies the first equation exactly. So
every positively homogeneous metric (8-hour mean maximum, peak) satisfies::

    metric(V, G, C_out, k, Q) = C_out + (k G / V) * M(Q / V),

where ``M(a)`` is the metric of the unit room. All four inputs collapse onto
**one curve of one variable**, the volume-specific flow ``a = Q / V``.
Consequences:

* ``G`` and ``k`` enter only as their product: a 10 % sensor over-count has
  exactly the same effect as a 10 % higher CO2 generation rate.
* ``Q_min`` solves ``M(a) = (limit - C_out) V / (k G)``, then ``Q_min = a V``.
  So it needs only an inversion of the tabulated ``M(a)``
  (:class:`ResponseCurve`), not a new set of ODE solves for every sample.

The curve is tabulated once with the exact solver. It is inverted by
piecewise-linear interpolation in log-log coordinates (Week 3), and the
result is checked against full root finding in the notebook.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .co2model import RoomParameters
from .design import CRITERIA, evaluate_ventilation

#: Unit room used to tabulate the scaled response ``M(a)``.
UNIT_ROOM: RoomParameters = RoomParameters(volume_m3=1.0, outdoor_co2_ppm=0.0, co2_generation_lps=1.0)

#: Supported distributions in :func:`sample_inputs` and their parameters.
DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "uniform": ("low", "high"),
    "loguniform": ("low", "high"),
    "triangular": ("low", "mode", "high"),
    "normal": ("mean", "sd"),
}


# ---------------------------------------------------------------------------
# Exact scaling law
# ---------------------------------------------------------------------------

def scaled_metric(
    unit_value: np.ndarray | float,
    volume_m3: np.ndarray | float,
    co2_generation_lps: np.ndarray | float,
    outdoor_co2_ppm: np.ndarray | float,
    count_factor: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Map a unit-room metric ``M(a)`` to a real room: ``C_out + (k G / V) M``.

    Valid for metrics that scale linearly with ``C - C_out``: the worst
    8-hour mean, the peak and the mean. It is **not** valid for "fraction of
    time above a limit". Arrays broadcast.
    """
    return np.asarray(outdoor_co2_ppm) + (np.asarray(count_factor) * np.asarray(co2_generation_lps)
                                         / np.asarray(volume_m3)) * np.asarray(unit_value)


@dataclass
class ResponseCurve:
    """Tabulated unit-room metrics ``M(a)`` for one occupancy record.

    Build it with :meth:`build`. Then :meth:`metric` and :meth:`q_min` give
    the worst 8-hour mean, the peak and ``Q_min`` for any ``(V, G, C_out, k)``
    without further ODE solves.

    Attributes
    ----------
    a : numpy.ndarray
        Volume-specific flows ``Q/V`` [(L/s)/m3], strictly increasing.
    values : dict of str to numpy.ndarray
        ``{"8h": M_8h(a), "peak": M_peak(a)}``: unit-room metrics, strictly
        decreasing in ``a``.
    """

    a: np.ndarray
    values: dict[str, np.ndarray]

    @classmethod
    def build(
        cls,
        occupancy: np.ndarray,
        dt_s: float,
        a_grid: np.ndarray | None = None,
    ) -> ResponseCurve:
        """Tabulate ``M(a)`` with one exact solve per grid point.

        Parameters
        ----------
        occupancy : array_like
            People per step (at least 8 hours, not all zero).
        dt_s : float
            Step length [s].
        a_grid : array_like, optional
            Flows ``Q/V`` [(L/s)/m3]. Defaults to 81 log-spaced values from
            0.05 to 10, i.e. 0.18–36 air changes per hour.

        Raises
        ------
        ValueError
            If the grid is not positive and strictly increasing, or the
            tabulated metrics are not strictly decreasing (for example, an
            empty room).
        """
        a = np.geomspace(0.05, 10.0, 81) if a_grid is None else np.asarray(a_grid, dtype=np.float64)
        if a.ndim != 1 or a.size < 2 or np.any(a <= 0) or np.any(np.diff(a) <= 0):
            raise ValueError("a_grid must be 1-D, positive and strictly increasing with >= 2 points")
        keys = {name: key for name, (key, _) in CRITERIA.items()}
        rows = [evaluate_ventilation(occupancy, float(ai), UNIT_ROOM, dt_s) for ai in a]
        values = {name: np.array([r[key] for r in rows]) for name, key in keys.items()}
        for name, v in values.items():
            if not (np.all(v > 0) and np.all(np.diff(v) < 0)):
                raise ValueError(f"tabulated {name} metric is not positive and strictly decreasing")
        return cls(a=a, values=values)

    def _check(self, criterion: str) -> np.ndarray:
        if criterion not in self.values:
            raise ValueError(f"criterion must be one of {sorted(self.values)}, got {criterion!r}")
        return self.values[criterion]

    def metric(
        self,
        criterion: str,
        ventilation_lps: np.ndarray | float,
        volume_m3: np.ndarray | float,
        co2_generation_lps: np.ndarray | float,
        outdoor_co2_ppm: np.ndarray | float,
        count_factor: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """Metric [ppm] for constant flow ``Q``, interpolating ``log M`` against ``log a``.

        Raises
        ------
        ValueError
            If any ``Q/V`` lies outside the tabulated range (no extrapolation).
        """
        v = self._check(criterion)
        a = np.asarray(ventilation_lps, dtype=np.float64) / np.asarray(volume_m3, dtype=np.float64)
        if np.any(a < self.a[0]) or np.any(a > self.a[-1]):
            raise ValueError(f"Q/V outside the tabulated range [{self.a[0]:.3g}, {self.a[-1]:.3g}]")
        m = np.exp(np.interp(np.log(a), np.log(self.a), np.log(v)))
        return scaled_metric(m, volume_m3, co2_generation_lps, outdoor_co2_ppm, count_factor)

    def q_min(
        self,
        criterion: str,
        volume_m3: np.ndarray | float,
        co2_generation_lps: np.ndarray | float,
        outdoor_co2_ppm: np.ndarray | float,
        count_factor: np.ndarray | float = 1.0,
        limit_ppm: float | None = None,
    ) -> np.ndarray:
        """Minimum constant flow ``Q_min`` [L/s] for each set of inputs (arrays broadcast).

        Solves ``M(a) = (limit - C_out) V / (k G)`` by inverse interpolation of
        the tabulated, strictly decreasing ``M(a)``, then ``Q_min = a V``.

        Raises
        ------
        ValueError
            If ``limit <= C_out`` for any sample, or the required ``M`` lies
            outside the tabulated range (no extrapolation).
        """
        v = self._check(criterion)
        limit = CRITERIA[criterion][1] if limit_ppm is None else float(limit_ppm)
        c_out = np.asarray(outdoor_co2_ppm, dtype=np.float64)
        if np.any(c_out >= limit):
            raise ValueError("the limit must exceed the outdoor CO2 concentration")
        target = (limit - c_out) * np.asarray(volume_m3) / (np.asarray(count_factor) * np.asarray(co2_generation_lps))
        if np.any(target > v[0]) or np.any(target < v[-1]):
            raise ValueError("required unit metric outside the tabulated range; widen a_grid")
        # np.interp needs increasing x: M is decreasing, so reverse both arrays.
        log_a = np.interp(np.log(target), np.log(v[::-1]), np.log(self.a[::-1]))
        return np.exp(log_a) * np.asarray(volume_m3, dtype=np.float64)


# ---------------------------------------------------------------------------
# One-at-a-time sensitivity
# ---------------------------------------------------------------------------

def oat_sensitivity(
    f: Callable[..., float],
    nominal: Mapping[str, float],
    ranges: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """One-at-a-time sensitivity of ``f`` over a low/high range for each input.

    Parameters
    ----------
    f : callable
        ``f(**inputs) -> float``. Called with every key of ``nominal``.
    nominal : mapping
        Nominal value of every input.
    ranges : mapping
        ``{name: (low, high)}`` for the inputs to vary (a subset of ``nominal``).

    Returns
    -------
    pandas.DataFrame
        One row per varied input, sorted by decreasing swing, with columns
        ``parameter``, ``low``, ``high``, ``f_low``, ``f_high``, ``f_nominal``
        and ``swing = |f_high - f_low|``. This is the data for a tornado chart.

    Raises
    ------
    KeyError
        If a range is given for an input that is not in ``nominal``.
    """
    unknown = set(ranges) - set(nominal)
    if unknown:
        raise KeyError(f"ranges given for unknown inputs: {sorted(unknown)}")
    f0 = float(f(**nominal))
    rows = []
    for name, (lo, hi) in ranges.items():
        f_lo = float(f(**{**nominal, name: lo}))
        f_hi = float(f(**{**nominal, name: hi}))
        rows.append({"parameter": name, "low": lo, "high": hi, "f_low": f_lo, "f_high": f_hi,
                     "f_nominal": f0, "swing": abs(f_hi - f_lo)})
    return pd.DataFrame(rows).sort_values("swing", ascending=False, kind="stable", ignore_index=True)


def elasticities(
    f: Callable[..., float],
    nominal: Mapping[str, float],
    names: list[str] | None = None,
    rel_step: float = 0.01,
) -> dict[str, float]:
    """Normalised local sensitivities ``(x / f) df/dx`` by central differences.

    An elasticity of ``e`` means a 1 % change in the input changes the output
    by about ``e`` %. Inputs with very different units can then be compared.
    For a power law ``f = c x^p`` the elasticity is ``p``; the central
    difference has an ``O(rel_step^2)`` error.

    Parameters
    ----------
    f : callable
        ``f(**inputs) -> float``.
    nominal : mapping
        Nominal inputs (non-zero for the inputs differentiated).
    names : list of str, optional
        Inputs to differentiate. Defaults to all of ``nominal``.
    rel_step : float, default 0.01
        Relative perturbation ``h``: inputs are set to ``x (1 +/- h)``.

    Raises
    ------
    ValueError
        If ``rel_step`` is not in (0, 0.5), or ``f`` or an input is zero at
        the nominal point.
    """
    if not 0 < rel_step < 0.5:
        raise ValueError("rel_step must be in (0, 0.5)")
    f0 = float(f(**nominal))
    if f0 == 0:
        raise ValueError("f is zero at the nominal point; elasticity undefined")
    out = {}
    for name in names or list(nominal):
        x = float(nominal[name])
        if x == 0:
            raise ValueError(f"input {name!r} is zero; elasticity undefined")
        f_up = float(f(**{**nominal, name: x * (1 + rel_step)}))
        f_dn = float(f(**{**nominal, name: x * (1 - rel_step)}))
        out[name] = (f_up - f_dn) / (2 * rel_step * f0)
    return out


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def sample_inputs(
    distributions: Mapping[str, tuple],
    n: int,
    seed: int | None = 0,
) -> dict[str, np.ndarray]:
    """Draw ``n`` independent samples of each input.

    Parameters
    ----------
    distributions : mapping
        ``{name: (kind, *params)}`` with ``kind`` one of
        ``("uniform", low, high)``, ``("loguniform", low, high)``,
        ``("triangular", low, mode, high)`` or ``("normal", mean, sd)``.
    n : int
        Number of samples (> 0).
    seed : int or None, default 0
        Seed of :func:`numpy.random.default_rng`, for reproducibility.

    Returns
    -------
    dict of str to numpy.ndarray

    Raises
    ------
    ValueError
        For an unknown distribution, the wrong number of parameters, or
        invalid parameters.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    rng = np.random.default_rng(seed)
    out = {}
    for name, spec in distributions.items():
        kind, *p = spec
        if kind not in DISTRIBUTIONS:
            raise ValueError(f"{name}: unknown distribution {kind!r}; use one of {sorted(DISTRIBUTIONS)}")
        if len(p) != len(DISTRIBUTIONS[kind]):
            raise ValueError(f"{name}: {kind} needs parameters {DISTRIBUTIONS[kind]}, got {p}")
        if kind in ("uniform", "loguniform") and not p[0] < p[1]:
            raise ValueError(f"{name}: need low < high")
        if kind == "loguniform":
            if p[0] <= 0:
                raise ValueError(f"{name}: loguniform needs low > 0")
            out[name] = np.exp(rng.uniform(np.log(p[0]), np.log(p[1]), n))
        elif kind == "uniform":
            out[name] = rng.uniform(p[0], p[1], n)
        elif kind == "triangular":
            if not p[0] <= p[1] <= p[2] or p[0] == p[2]:
                raise ValueError(f"{name}: need low <= mode <= high and low < high")
            out[name] = rng.triangular(p[0], p[1], p[2], n)
        else:
            if p[1] <= 0:
                raise ValueError(f"{name}: normal needs sd > 0")
            out[name] = rng.normal(p[0], p[1], n)
    return out


def percentile_summary(values: np.ndarray, percentiles: tuple[float, ...] = (5, 50, 95)) -> dict[str, float]:
    """Mean, standard deviation, standard error of the mean and percentiles of a sample.

    The standard error ``sd / sqrt(n)`` shows the ``1/sqrt(n)`` convergence of
    Monte Carlo estimates.

    Raises
    ------
    ValueError
        If ``values`` is empty or contains non-finite entries.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("values must be a non-empty array of finite numbers")
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    out = {"n": float(v.size), "mean": float(v.mean()), "sd": sd, "sem": sd / np.sqrt(v.size)}
    out.update({f"p{p:g}": float(np.percentile(v, p)) for p in percentiles})
    return out


# ---------------------------------------------------------------------------
# Numerical sensitivity helper
# ---------------------------------------------------------------------------

def aggregate_occupancy(occupancy: np.ndarray, factor: int) -> np.ndarray:
    """Average consecutive blocks of ``factor`` steps (a coarser grid).

    Each 1-minute value is already an exact time-average, so the block mean
    is the exact time-average over the coarser bin, and person-time is
    conserved. An incomplete final block is dropped.

    Raises
    ------
    ValueError
        If ``factor`` is not a positive integer or exceeds the series length.
    """
    occ = np.asarray(occupancy, dtype=np.float64)
    if not isinstance(factor, (int, np.integer)) or not 1 <= factor <= occ.size:
        raise ValueError(f"factor must be an integer in [1, {occ.size}], got {factor!r}")
    n = occ.size // factor
    return occ[: n * factor].reshape(n, factor).mean(axis=1)
