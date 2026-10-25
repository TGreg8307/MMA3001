"""Engineering output metrics computed from a simulated CO2 series.

These are the "decision-relevant outputs" of the project. They are what a
building engineer acts on, and so what convergence and sensitivity studies
should report (MMA3001 Notebook 7.3.17):

* the **8-hour time-weighted mean** CO2, compared with the NCC
  verification-method limit of 850 ppm;
* the **peak** concentration and the **fraction of time above 1000 ppm**;
* the overall mean concentration.

All functions take *step means*: the exact time-average of C over each
step, from :func:`occvent.co2model.step_mean_concentration`. A moving
average of step means is then the exact time-average over the window.
"""

from __future__ import annotations

import numpy as np

#: NCC verification-method limit for CO2: 850 ppm averaged over 8 hours.
NCC_8H_LIMIT_PPM: float = 850.0
#: Secondary short-term criterion used in this project [ppm].
PEAK_LIMIT_PPM: float = 1000.0
#: Averaging window of the NCC limit [s].
EIGHT_HOURS_S: float = 8 * 3600.0


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average over ``window`` consecutive steps.

    Computed with a cumulative sum in O(n) operations. Element ``i`` is the
    mean of ``values[i - window + 1 : i + 1]``. The first ``window - 1``
    elements are ``NaN`` because a full window is not yet available.

    Parameters
    ----------
    values : array_like
        1-D series of finite values (e.g. step-mean CO2 [ppm]).
    window : int
        Number of steps in the window, ``1 <= window <= len(values)``.

    Returns
    -------
    numpy.ndarray
        Moving average, same length as ``values``.

    Raises
    ------
    ValueError
        If ``values`` is not 1-D, contains non-finite values, or ``window``
        is out of range.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("values must be 1-D")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")
    if not isinstance(window, (int, np.integer)) or not 1 <= window <= v.size:
        raise ValueError(f"window must be an integer in [1, {v.size}], got {window!r}")
    # Subtract the first value before summing to reduce round-off in long sums.
    shift = v[0]
    csum = np.concatenate(([0.0], np.cumsum(v - shift)))
    out = np.full(v.size, np.nan)
    out[window - 1:] = (csum[window:] - csum[:-window]) / window + shift
    return out


def compliance_metrics(
    step_mean_ppm: np.ndarray,
    dt_s: float,
    limit_8h_ppm: float = NCC_8H_LIMIT_PPM,
    peak_limit_ppm: float = PEAK_LIMIT_PPM,
) -> dict[str, float]:
    """Summarise a CO2 series against the indoor-air-quality criteria.

    Parameters
    ----------
    step_mean_ppm : array_like
        Exact step-mean CO2 [ppm] on a regular grid.
    dt_s : float
        Step length [s]. Must divide 8 hours exactly (e.g. 60 s).
    limit_8h_ppm : float, default 850
        Limit on the 8-hour mean [ppm].
    peak_limit_ppm : float, default 1000
        Short-term threshold [ppm].

    Returns
    -------
    dict
        * ``max_8h_mean_ppm``: highest 8-hour moving average.
        * ``exceeds_8h_limit``: 1.0 if it is above ``limit_8h_ppm``, else 0.0.
        * ``margin_8h_ppm``: ``limit_8h_ppm - max_8h_mean_ppm`` (negative
          means non-compliant).
        * ``peak_ppm``: largest step mean.
        * ``fraction_above_peak_limit``: share of time above ``peak_limit_ppm``.
        * ``mean_ppm``: overall time-average.

    Raises
    ------
    ValueError
        If ``dt_s`` does not divide 8 hours, or the series is shorter than 8 hours.
    """
    v = np.asarray(step_mean_ppm, dtype=np.float64)
    steps = EIGHT_HOURS_S / dt_s
    if dt_s <= 0 or abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"dt_s={dt_s} must divide 8 hours exactly")
    window = int(round(steps))
    if v.size < window:
        raise ValueError(f"series covers {v.size * dt_s / 3600:.2f} h; at least 8 h is required")
    r8 = rolling_mean(v, window)
    max8 = float(np.nanmax(r8))
    return {
        "max_8h_mean_ppm": max8,
        "exceeds_8h_limit": float(max8 > limit_8h_ppm),
        "margin_8h_ppm": float(limit_8h_ppm - max8),
        "peak_ppm": float(v.max()),
        "fraction_above_peak_limit": float(np.mean(v > peak_limit_ppm)),
        "mean_ppm": float(v.mean()),
    }
