"""Validation, de-duplication and regular-grid resampling of occupancy events.

The sensor is *event driven*: it writes a row when the head count changes,
not at fixed intervals. Between events the count is taken to be constant
(a zero-order hold). The physics model needs N(t) on a regular time grid,
so :func:`resample_occupancy` integrates the piecewise-constant signal
exactly over each grid bin. This keeps person-time (person-seconds), and
therefore the CO2 generated, conserved through resampling.

A hold cannot be trusted forever, because a silent sensor could mean
"no change" or "offline". The data audit found:

* Occupied holds (count > 0) in G.20 never exceeded 2.0 h. Longer occupied
  holds (up to 9.5 h) occur only in G.25 and are consistent with a stuck
  count.
* Vacant holds (count = 0) last up to 87 h, covering nights, weekends and
  long weekends, where an empty room is expected.

So each hold is trusted for at most ``max_occupied_hold`` (default 2.5 h)
when occupied, or ``max_vacant_hold`` (default 96 h) when vacant. Time
beyond that limit is treated as unobserved. Both limits are parameters so
their influence can be tested in the sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import (
    LOCAL_TIMEZONE,
    OCCUPIED_STATUS,
    RAW_COLUMNS,
    TIMESTAMP_FORMAT,
    VALID_STATUSES,
)


@dataclass(frozen=True)
class ResampleSettings:
    """Parameters of :func:`resample_occupancy`.

    Attributes
    ----------
    step : str
        Grid spacing as a pandas timedelta string, e.g. ``"1min"``. Must be
        a whole number of seconds.
    max_occupied_hold : str
        Longest time a non-zero count is held without a new event.
    max_vacant_hold : str
        Longest time a zero count is held without a new event.
    min_coverage : float
        Minimum observed fraction of a bin for it to be marked ``valid``,
        in (0, 1]. The default 1.0 marks only fully observed bins as valid.
    """

    step: str = "1min"
    max_occupied_hold: str = "2.5h"
    max_vacant_hold: str = "96h"
    min_coverage: float = 1.0

    def as_dict(self) -> dict:
        """Return the settings as a plain ``dict`` (for JSON reports)."""
        return asdict(self)


def _whole_seconds(value: str | pd.Timedelta, name: str) -> int:
    """Convert a timedelta-like value to a positive whole number of seconds.

    Raises
    ------
    ValueError
        If the value cannot be parsed, is not positive, or is not a whole
        number of seconds.
    """
    try:
        td = pd.Timedelta(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}={value!r} is not a valid time interval") from exc
    seconds = td.total_seconds()
    if seconds <= 0 or seconds != int(seconds):
        raise ValueError(f"{name} must be a positive whole number of seconds, got {value!r}")
    return int(seconds)


def clean_events(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate, sort and de-duplicate the raw events of a single room.

    Rows are checked in this order. Each rejected row is counted once,
    under the first rule it fails:

    1. ``collecteddate`` must parse as ``YYYY-MM-DD HH:MM:SS+HH``.
    2. ``headcount`` must be a finite, non-negative integer.
    3. ``occupancystatus`` must be one of :data:`occvent.config.VALID_STATUSES`.
    4. ``id`` must be present (it breaks ties between equal timestamps).
    5. Status and count must agree: ``CurrentlyOccupied`` if and only if
       ``headcount > 0``.

    Surviving rows are converted to UTC and sorted by ``(t, id)``. Where
    several rows share a timestamp, the one with the largest ``id`` (the
    last written) is kept.

    Parameters
    ----------
    raw : pandas.DataFrame
        Rows for one ``floorspaceid``, with the columns in
        :data:`occvent.config.RAW_COLUMNS` (as returned by
        :func:`occvent.io.read_occupancy_events`).

    Returns
    -------
    events : pandas.DataFrame
        Columns ``t`` (tz-aware UTC timestamp), ``headcount`` (int64, people),
        ``status`` (str) and ``id`` (int64). Strictly increasing in ``t``.
    report : dict
        Row counts for each rejection rule, duplicates removed, rows out of
        chronological order in the file, and the first and last event times.

    Raises
    ------
    ValueError
        If required columns are missing or the rows span more than one
        ``floorspaceid``.
    """
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"raw is missing required column(s): {missing}")
    spaces = raw["floorspaceid"].dropna().unique()
    if len(spaces) > 1:
        raise ValueError(f"clean_events expects one floorspaceid, got {len(spaces)}")

    report: dict = {"rows_in": int(len(raw))}

    t = pd.to_datetime(raw["collecteddate"], format=TIMESTAMP_FORMAT, utc=True, errors="coerce")
    hc = pd.to_numeric(raw["headcount"], errors="coerce").astype("float64")
    rid = pd.to_numeric(raw["id"], errors="coerce").astype("float64")
    status = raw["occupancystatus"].astype("str")

    ok = t.notna().to_numpy()
    report["rejected_bad_timestamp"] = int((~ok).sum())

    hc_vals = hc.to_numpy()
    with np.errstate(invalid="ignore"):
        hc_ok = np.isfinite(hc_vals) & (hc_vals >= 0) & (hc_vals == np.floor(hc_vals))
    report["rejected_bad_headcount"] = int((ok & ~hc_ok).sum())
    ok = ok & hc_ok

    st_ok = status.isin(VALID_STATUSES).to_numpy() & raw["occupancystatus"].notna().to_numpy()
    report["rejected_bad_status"] = int((ok & ~st_ok).sum())
    ok = ok & st_ok

    id_ok = np.isfinite(rid.to_numpy())
    report["rejected_missing_id"] = int((ok & ~id_ok).sum())
    ok = ok & id_ok

    with np.errstate(invalid="ignore"):
        consistent = (status.to_numpy() == OCCUPIED_STATUS) == (hc_vals > 0)
    report["rejected_status_count_mismatch"] = int((ok & ~consistent).sum())
    ok = ok & consistent

    events = pd.DataFrame(
        {
            "t": t.loc[ok].reset_index(drop=True),  # stays tz-aware (UTC)
            "headcount": hc_vals[ok].astype("int64"),
            "status": status.to_numpy()[ok],
            "id": rid.to_numpy()[ok].astype("int64"),
        }
    )
    report["rows_out_of_order_in_file"] = int((events["t"].diff() < pd.Timedelta(0)).sum())

    events = events.sort_values(["t", "id"], kind="mergesort").reset_index(drop=True)
    dup = events["t"].duplicated(keep="last")
    report["duplicates_removed"] = int(dup.sum())
    events = events.loc[~dup].reset_index(drop=True)

    report["rows_out"] = int(len(events))
    report["first_event_utc"] = events["t"].iloc[0].isoformat() if len(events) else None
    report["last_event_utc"] = events["t"].iloc[-1].isoformat() if len(events) else None
    return events, report


def resample_occupancy(
    events: pd.DataFrame,
    settings: ResampleSettings | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Convert an event sequence to a regular, time-weighted occupancy series.

    Each event value ``n_i`` is held from its time ``t_i`` until the next
    event or until the hold limit, whichever comes first. The bin value is
    the exact time-average of this step function over the observed part of
    the bin::

        N_bin = (integral of N over the observed part) / (observed duration)

    The integral is evaluated at all bin edges at once from cumulative sums
    (O(n log n), fully vectorised), using integer seconds so the arithmetic
    is exact.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of :func:`clean_events`. Columns ``t`` (tz-aware, strictly
        increasing, whole seconds) and ``headcount`` (non-negative integers).
    settings : ResampleSettings, optional
        Grid step, hold limits and validity threshold. Defaults to
        :class:`ResampleSettings()`.

    Returns
    -------
    grid : pandas.DataFrame
        Indexed by bin start (tz-aware UTC ``DatetimeIndex``, name ``t_utc``),
        with columns:

        * ``headcount``: float, time-averaged people in the bin. ``NaN`` if
          no part of the bin was observed.
        * ``coverage``: float in [0, 1], the observed fraction of the bin.
        * ``valid``: bool, ``coverage >= settings.min_coverage``.
    report : dict
        Settings used, bin counts, number of holds cut short by each limit,
        total unobserved hours, and the longest occupied and vacant holds.

    Raises
    ------
    ValueError
        If ``events`` is empty, is not strictly increasing in time, has
        timestamps that are not whole seconds or not tz-aware, or if any
        setting is invalid.
    """
    settings = settings or ResampleSettings()
    step_s = _whole_seconds(settings.step, "step")
    occ_s = _whole_seconds(settings.max_occupied_hold, "max_occupied_hold")
    vac_s = _whole_seconds(settings.max_vacant_hold, "max_vacant_hold")
    if not 0.0 < settings.min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in (0, 1], got {settings.min_coverage}")
    if events.empty:
        raise ValueError("events is empty; nothing to resample")
    if events["t"].dt.tz is None:
        raise ValueError("events['t'] must be tz-aware")
    t = events["t"].dt.tz_convert("UTC")
    if not t.is_monotonic_increasing or t.duplicated().any():
        raise ValueError("events must be strictly increasing in time; run clean_events first")
    n = events["headcount"].to_numpy()
    if np.any(n < 0) or not np.all(n == np.floor(n)):
        raise ValueError("headcount must contain non-negative integers")
    n = n.astype(np.int64)

    step_td = pd.Timedelta(seconds=step_s)
    start = t.iloc[0].floor(step_td)
    end = t.iloc[-1].ceil(step_td)
    if end == start:  # a single event exactly on a bin edge
        end = start + step_td

    # Event times as integer seconds from the grid origin.
    s_float = ((t - start) / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)
    if not np.all(s_float == np.round(s_float)):
        raise ValueError("event timestamps must be whole seconds")
    s = s_float.astype(np.int64)

    # Segment i: value n[i] on [s[i], e[i]), where e[i] = min(next event, hold limit).
    nxt = np.empty_like(s)
    nxt[:-1] = s[1:]
    nxt[-1] = s[-1]  # nothing is known after the last event
    limit = np.where(n > 0, occ_s, vac_s)
    e = np.minimum(nxt, s + limit)
    seglen = e - s
    gap = nxt - s
    truncated = gap > limit

    # Cumulative person-seconds (A) and observed seconds (L) before each segment.
    cum_a = np.concatenate(([0], np.cumsum(n * seglen)[:-1]))
    cum_l = np.concatenate(([0], np.cumsum(seglen)[:-1]))

    n_bins = int((end - start) // step_td)
    x = np.arange(n_bins + 1, dtype=np.int64) * step_s  # bin edges
    k = np.searchsorted(s, x, side="right") - 1  # segment containing each edge
    before_first = k < 0
    kk = np.where(before_first, 0, k)
    within = np.clip(x - s[kk], 0, seglen[kk])
    big_f = np.where(before_first, 0, cum_a[kk] + n[kk] * within)
    big_g = np.where(before_first, 0, cum_l[kk] + within)

    person_s = np.diff(big_f)
    observed_s = np.diff(big_g)
    coverage = observed_s / step_s
    headcount = np.full(n_bins, np.nan)
    seen = observed_s > 0
    headcount[seen] = person_s[seen] / observed_s[seen]

    index = pd.date_range(start, periods=n_bins, freq=step_td, name="t_utc")
    grid = pd.DataFrame(
        {"headcount": headcount, "coverage": coverage, "valid": coverage >= settings.min_coverage},
        index=index,
    )

    occ = n > 0
    report = {
        "settings": settings.as_dict(),
        "n_bins": n_bins,
        "n_valid_bins": int(grid["valid"].sum()),
        "fraction_valid": float(grid["valid"].mean()),
        "occupied_holds_truncated": int((truncated & occ).sum()),
        "vacant_holds_truncated": int((truncated & ~occ).sum()),
        "unobserved_hours_from_truncation": float((gap - seglen)[truncated].sum() / 3600.0),
        "longest_occupied_hold_hours": float(gap[occ].max() / 3600.0) if occ.any() else 0.0,
        "longest_vacant_hold_hours": float(gap[~occ].max() / 3600.0) if (~occ).any() else 0.0,
        "person_hours_observed": float(person_s.sum() / 3600.0),
    }
    return grid, report


def add_quality_flags(
    grid: pd.DataFrame,
    tz: str = LOCAL_TIMEZONE,
    night_hours: tuple[int, int] = (1, 5),
    capacity: float | None = None,
) -> pd.DataFrame:
    """Add boolean data-quality flags to a resampled series.

    Flags mark suspicious values; they do not remove them. Later analyses
    decide whether to exclude, keep or test sensitivity to flagged data.

    Parameters
    ----------
    grid : pandas.DataFrame
        Output of :func:`resample_occupancy` (tz-aware index, ``headcount``
        column).
    tz : str, default :data:`occvent.config.LOCAL_TIMEZONE`
        Time zone used to decide whether a bin falls at night.
    night_hours : tuple of int, default (1, 5)
        Local-time window ``[start_hour, end_hour)`` when the building is
        expected to be empty. Requires ``0 <= start_hour < end_hour <= 24``.
    capacity : float, optional
        Plausible maximum number of occupants. If given, bins above it are
        flagged. If ``None``, no bin is flagged ``over_capacity``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``grid`` with columns ``night_occupied`` (headcount > 0
        during night hours) and ``over_capacity`` added.

    Raises
    ------
    ValueError
        If ``night_hours`` or ``capacity`` is out of range, or the index is
        not tz-aware.
    """
    h0, h1 = night_hours
    if not (0 <= h0 < h1 <= 24):
        raise ValueError(f"night_hours must satisfy 0 <= start < end <= 24, got {night_hours}")
    if capacity is not None and capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if not isinstance(grid.index, pd.DatetimeIndex) or grid.index.tz is None:
        raise ValueError("grid must have a tz-aware DatetimeIndex")

    out = grid.copy()
    hour = out.index.tz_convert(tz).hour
    is_night = (hour >= h0) & (hour < h1)
    occupied = out["headcount"].fillna(0).to_numpy() > 0
    out["night_occupied"] = is_night & occupied
    if capacity is None:
        out["over_capacity"] = False
    else:
        out["over_capacity"] = out["headcount"].fillna(0).to_numpy() > capacity
    return out
