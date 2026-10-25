"""Day-ahead occupancy forecasting (machine-learning extension).

Prediction contract
-------------------
* **Issued:** at local midnight, at the start of the target day.
* **Target:** the mean occupancy of each 15-minute slot of that day (96 values;
  92 or 100 on daylight-saving change days).
* **Allowed information:** the calendar (time of day, weekday, Victorian
  public holidays) and occupancy measured on **earlier days only**.
* **Use:** schedule the room's ventilation for the day ahead, without relying
  on a live sensor.

Leakage is prevented by construction. All occupancy features come from
earlier *local calendar dates* (:func:`daily_slot_table` and
:func:`make_features`), so no value from the target day can enter a
feature, including on daylight-saving change days.

Teaching-period (semester) dates are **not** used as a feature. Official
Monash dates for 2024–25 could not be verified, so the model learns term
changes from recent occupancy instead (the 1-day and 7-day lags and the
4-week mean). This means it reacts to the start of a term with a delay of
up to a week, which is a stated limitation.

Models are those taught in MMA3001 Week 5: a baseline, linear regression,
a decision tree and (optionally) a neural network, each built as a
scikit-learn ``Pipeline`` so that every learned preprocessing step is
fitted on training data only (Notebook 5.6.8).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import LOCAL_TIMEZONE

#: Victorian public holidays in the study period (Business Victoria, "Public
#: holidays in Victoria for 2024 / 2025 / 2026", accessed 25 Oct 2026).
VIC_PUBLIC_HOLIDAYS: frozenset[dt.date] = frozenset(dt.date.fromisoformat(d) for d in (
    "2024-01-01", "2024-01-26", "2024-03-11", "2024-03-29", "2024-03-30", "2024-03-31", "2024-04-01",
    "2024-04-25", "2024-06-10", "2024-09-27", "2024-11-05", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-01-27", "2025-03-10", "2025-04-18", "2025-04-19", "2025-04-20", "2025-04-21",
    "2025-04-25", "2025-06-09", "2025-09-26", "2025-11-04", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-01-26", "2026-03-09", "2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-04-25", "2026-06-08", "2026-09-25", "2026-11-03", "2026-12-25", "2026-12-26", "2026-12-28",
))

#: Number of daily harmonics (Fourier terms) used for the time of day.
N_HARMONICS: int = 3

#: Feature columns produced by :func:`make_features`, in order.
FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"tod_{f}{k}" for k in range(1, N_HARMONICS + 1) for f in ("sin", "cos")),
    *(f"dow_{d}" for d in range(7)),
    "holiday", "workday",
    *(f"work_tod_{f}{k}" for k in range(1, N_HARMONICS + 1) for f in ("sin", "cos")),
    "lag_1d", "lag_7d", "lag_mean_4w",
    "lag_1d_2h", "lag_7d_2h", "lag_mean_4w_2h",
    "prev_day_mean", "prev_week_mean",
)

#: Width of the centred smoothing window for the ``*_2h`` lag features [slots].
SMOOTH_SLOTS: int = 9


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def daily_slot_table(
    occupancy: pd.Series,
    slot_minutes: int = 15,
    tz: str = LOCAL_TIMEZONE,
) -> pd.DataFrame:
    """Reshape a regular occupancy series into a (local date x slot of day) table.

    Each cell is the mean occupancy over that local clock slot, which is the
    exact time-average when the input values are themselves exact averages
    over equal steps. On the autumn daylight-saving day, the repeated hour is
    averaged. On the spring day, the skipped hour is ``NaN``.

    Parameters
    ----------
    occupancy : pandas.Series
        People per step, with a regular, tz-aware ``DatetimeIndex`` whose step
        divides ``slot_minutes``.
    slot_minutes : int, default 15
        Slot length; must divide 60 evenly.
    tz : str
        Local time zone defining dates and clock slots.

    Returns
    -------
    pandas.DataFrame
        Index: local dates (``datetime.date``). Columns: slot numbers
        ``0 .. 24*60/slot_minutes - 1``.

    Raises
    ------
    ValueError
        If the index is not tz-aware or ``slot_minutes`` does not divide 60.
    """
    if not isinstance(occupancy.index, pd.DatetimeIndex) or occupancy.index.tz is None:
        raise ValueError("occupancy needs a tz-aware DatetimeIndex")
    if slot_minutes <= 0 or 60 % slot_minutes:
        raise ValueError("slot_minutes must be a positive divisor of 60")
    local = occupancy.index.tz_convert(tz)
    slot = (local.hour * 60 + local.minute) // slot_minutes
    frame = pd.DataFrame({"date": local.date, "slot": slot, "n": occupancy.to_numpy(dtype=np.float64)})
    table = frame.groupby(["date", "slot"])["n"].mean().unstack("slot")
    return table.reindex(columns=range(24 * 60 // slot_minutes))


def fill_short_gaps(table: pd.DataFrame, max_slots: int = 4) -> pd.DataFrame:
    """Linearly interpolate runs of at most ``max_slots`` missing slots inside each day.

    This is used **only for lag features**. On the spring daylight-saving day
    the 02:00–03:00 hour does not exist, so a "same slot yesterday" lag would
    be missing for that hour on the following days. Interpolating between the
    neighbouring slots of the *earlier* day keeps those rows usable without
    using any later information. Longer gaps, and gaps at the start or end of
    a day, are left missing. The target itself is never filled.
    """
    a = table.to_numpy(dtype=np.float64, copy=True)
    n = a.shape[1]
    for row in a:
        missing = np.isnan(row)
        if not missing.any() or missing.all():
            continue
        j = 0
        while j < n:
            if not missing[j]:
                j += 1
                continue
            k = j
            while k < n and missing[k]:
                k += 1
            if k - j <= max_slots and j > 0 and k < n:
                row[j:k] = np.interp(np.arange(j, k), [j - 1, k], [row[j - 1], row[k]])
            j = k
    return pd.DataFrame(a, index=table.index, columns=table.columns)


def make_features(
    table: pd.DataFrame,
    holidays: frozenset[dt.date] = VIC_PUBLIC_HOLIDAYS,
    n_harmonics: int = N_HARMONICS,
) -> pd.DataFrame:
    """Build the day-ahead feature matrix and target from a daily slot table.

    Every occupancy feature uses **earlier dates only** (no leakage):

    * ``lag_1d``, ``lag_7d``: same slot 1 and 7 days earlier;
    * ``lag_mean_4w``: mean of the same slot 7, 14, 21 and 28 days earlier;
    * ``lag_1d_2h``, ``lag_7d_2h``, ``lag_mean_4w_2h``: the same lags of a
      centred 2-hour (9-slot) moving average *within each earlier day*. They
      are less noisy than single-slot values;
    * ``prev_day_mean``, ``prev_week_mean``: mean occupancy over the previous
      day and the previous 7 days.

    Short within-day gaps (the hour skipped at the spring daylight-saving
    change) are interpolated for the lag features only (:func:`fill_short_gaps`).

    Calendar features: ``n_harmonics`` daily sine/cosine pairs of the time of
    day, one-hot weekday, ``holiday``, ``workday`` (Mon–Fri and not a
    holiday), and the time-of-day harmonics multiplied by ``workday``. These
    products let a *linear* model use a different daily shape on work days.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`daily_slot_table`. Dates must be consecutive (missing
        dates are inserted as ``NaN`` rows, so lags stay aligned to calendar days).
    holidays : set of datetime.date
        Public holidays.
    n_harmonics : int
        Number of daily harmonics; :data:`FEATURE_COLUMNS` assumes the default.

    Returns
    -------
    pandas.DataFrame
        One row per (date, slot) with all features known and a finite target.
        Index: ``MultiIndex`` (``date``, ``slot``). Columns:
        :data:`FEATURE_COLUMNS` (for the default ``n_harmonics``) and ``target``.
    """
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")
    dates = pd.date_range(min(table.index), max(table.index), freq="D").date
    target = table.reindex(dates)
    t = fill_short_gaps(target)          # for lag features only (see fill_short_gaps)
    n_slots = t.shape[1]

    lag1 = t.shift(1)
    lag7 = t.shift(7)
    mean4w = sum(t.shift(7 * w) for w in range(1, 5)) / 4.0
    # Smooth along the slots of each day (rows); windows never cross into another date.
    smooth = t.T.rolling(SMOOTH_SLOTS, center=True, min_periods=1).mean().T
    smooth1, smooth7 = smooth.shift(1), smooth.shift(7)
    smooth4w = sum(smooth.shift(7 * w) for w in range(1, 5)) / 4.0
    prev_day = t.mean(axis=1, skipna=True).shift(1)
    prev_week = t.mean(axis=1, skipna=True).rolling(7, min_periods=7).mean().shift(1)

    def long(df: pd.DataFrame, name: str) -> pd.Series:
        s = df.stack(future_stack=True)
        s.index = s.index.set_names(["date", "slot"])
        return s.rename(name)

    out = pd.concat([long(target, "target"), long(lag1, "lag_1d"), long(lag7, "lag_7d"),
                     long(mean4w, "lag_mean_4w"), long(smooth1, "lag_1d_2h"),
                     long(smooth7, "lag_7d_2h"), long(smooth4w, "lag_mean_4w_2h")], axis=1)
    d = out.index.get_level_values("date")
    slot = out.index.get_level_values("slot").to_numpy()
    dow = np.array([x.weekday() for x in d])
    holiday = np.array([x in holidays for x in d], dtype=float)
    workday = ((dow < 5) & (holiday == 0)).astype(float)
    phase = 2 * np.pi * slot / n_slots
    for k in range(1, n_harmonics + 1):
        out[f"tod_sin{k}"] = np.sin(k * phase)
        out[f"tod_cos{k}"] = np.cos(k * phase)
    for day in range(7):
        out[f"dow_{day}"] = (dow == day).astype(float)
    out["holiday"] = holiday
    out["workday"] = workday
    for k in range(1, n_harmonics + 1):
        out[f"work_tod_sin{k}"] = workday * out[f"tod_sin{k}"]
        out[f"work_tod_cos{k}"] = workday * out[f"tod_cos{k}"]
    out["prev_day_mean"] = prev_day.reindex(d).to_numpy()
    out["prev_week_mean"] = prev_week.reindex(d).to_numpy()

    columns = [c for c in FEATURE_COLUMNS if c in out.columns]
    out = out[columns + ["target"]]
    return out[np.isfinite(out.to_numpy()).all(axis=1)]


def period_mask(index: pd.MultiIndex, start: str, end: str) -> np.ndarray:
    """Boolean mask of rows whose ``date`` level lies in ``[start, end]`` (inclusive, ISO dates)."""
    d = pd.to_datetime(pd.Index(index.get_level_values("date")))
    return np.asarray((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LagBaseline(RegressorMixin, BaseEstimator):
    """Naive baseline that predicts one lag feature unchanged (no parameters are learned).

    ``LagBaseline("lag_7d")`` is the seasonal-naive forecast "same time last
    week"; ``LagBaseline("lag_mean_4w")`` averages the same slot over the last
    four weeks. Being a scikit-learn estimator, it can be cross-validated
    exactly like the real models.

    Parameters
    ----------
    column : str
        Feature column to return as the prediction.
    """

    def __init__(self, column: str = "lag_7d"):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D102 - nothing to learn
        if self.column not in X.columns:
            raise ValueError(f"column {self.column!r} not in X")
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: D102
        return X[self.column].to_numpy(dtype=np.float64)


def build_models(
    tree_depth: int = 8,
    tree_min_leaf: int = 50,
    mlp_hidden: tuple[int, ...] = (32, 16),
    mlp_alpha: float = 1e-3,
    random_state: int = 0,
) -> dict[str, Pipeline]:
    """The candidate models, each an unfitted scikit-learn ``Pipeline``.

    * ``"linear"``: standardisation + ordinary least squares (5.1).
    * ``"tree"``: a regression tree with limited depth and minimum leaf size (5.2).
      No scaling is needed, because splits are invariant to monotone rescaling.
    * ``"mlp"``: standardisation + a multilayer perceptron with early
      stopping and a fixed seed (5.4). Scaling sits inside the pipeline, so it
      is learned from training rows only.
    """
    return {
        "linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "tree": Pipeline([("model", DecisionTreeRegressor(max_depth=tree_depth, min_samples_leaf=tree_min_leaf,
                                                           random_state=random_state))]),
        "mlp": Pipeline([("scale", StandardScaler()),
                         ("model", MLPRegressor(hidden_layer_sizes=mlp_hidden, alpha=mlp_alpha,
                                                early_stopping=True, max_iter=300, n_iter_no_change=15,
                                                random_state=random_state))]),
    }


def predict_nonnegative(model, X: pd.DataFrame) -> np.ndarray:
    """Model prediction clipped at zero: occupancy cannot be negative (a physical constraint, 5.4.13)."""
    return np.maximum(np.asarray(model.predict(X), dtype=np.float64), 0.0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, R² and bias (mean of ``y_pred - y_true``; negative = under-prediction).

    Raises
    ------
    ValueError
        If the arrays are empty, differ in length or contain non-finite values.
    """
    y, p = np.asarray(y_true, dtype=np.float64).ravel(), np.asarray(y_pred, dtype=np.float64).ravel()
    if y.size == 0 or y.size != p.size or not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        raise ValueError("need two finite arrays of equal, non-zero length")
    e = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))),
            "R2": float(1 - np.sum(e**2) / ss_tot) if ss_tot > 0 else float("nan"),
            "bias": float(np.mean(e))}


def bootstrap_mae_by_day(
    abs_errors: np.ndarray,
    dates: np.ndarray,
    n_boot: int = 2000,
    percentiles: tuple[float, float] = (2.5, 97.5),
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for the MAE by resampling whole **days** with replacement.

    Errors within a day are correlated, so resampling individual slots would
    understate the variability. Resampling days keeps each day's errors
    together (a block bootstrap). As in Notebook 5.5.11, this describes the
    precision of one test MAE for a fixed model; it is not a prediction interval.

    Raises
    ------
    ValueError
        If the inputs differ in length, are empty, or ``n_boot < 1``.
    """
    a = np.asarray(abs_errors, dtype=np.float64)
    d = np.asarray(dates)
    if a.size == 0 or a.size != d.size or n_boot < 1:
        raise ValueError("abs_errors and dates must be non-empty and equal length; n_boot >= 1")
    _, codes = np.unique(d, return_inverse=True)
    sums = np.bincount(codes, weights=a)
    counts = np.bincount(codes).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, sums.size, size=(n_boot, sums.size))
    maes = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.percentile(maes, percentiles)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Back to the physics model
# ---------------------------------------------------------------------------

def slots_to_steps(
    slot_values: pd.Series,
    step_index: pd.DatetimeIndex,
    slot_minutes: int = 15,
    tz: str = LOCAL_TIMEZONE,
) -> np.ndarray:
    """Expand per-(date, slot) values onto a finer time grid (zero-order hold).

    Each step takes the value of the local slot that contains it. Holding a
    value constant within its slot conserves person-time, as in Stage 1.

    Parameters
    ----------
    slot_values : pandas.Series
        Indexed by (``date``, ``slot``), as produced by :func:`make_features`.
    step_index : pandas.DatetimeIndex
        tz-aware start times of the fine steps (e.g. the 1-minute grid).

    Returns
    -------
    numpy.ndarray
        One value per step; ``NaN`` where no slot value exists.
    """
    if step_index.tz is None:
        raise ValueError("step_index must be tz-aware")
    local = step_index.tz_convert(tz)
    key = pd.MultiIndex.from_arrays([local.date, (local.hour * 60 + local.minute) // slot_minutes],
                                    names=["date", "slot"])
    s = slot_values.groupby(level=["date", "slot"]).mean()   # one value per key
    return s.reindex(key).to_numpy(dtype=np.float64)
