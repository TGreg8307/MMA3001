"""Reading the raw occupancy export and the processed per-room files.

The raw export is a single CSV of about 1.6 GB, roughly 9.1 million rows
(it may also be passed as the ``.zip`` it was supplied in). It is read in
chunks so that peak memory scales with the rows kept, not with the file
size.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .config import RAW_COLUMNS

#: Index name of processed per-room time series.
PROCESSED_INDEX: str = "t_utc"


def read_occupancy_events(
    path: str | Path,
    floorspace_ids: Iterable[str],
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Read the raw occupancy export, keeping only the requested spaces.

    No values are validated here; that is done by
    :func:`occvent.cleaning.clean_events` so that every rejected row is
    counted in its report. Numeric columns that cannot be parsed become
    ``NaN`` instead of raising.

    Parameters
    ----------
    path : str or pathlib.Path
        Raw export: a ``.csv`` file, or a ``.zip`` containing exactly one CSV.
    floorspace_ids : iterable of str
        ``floorspaceid`` values to keep. Must not be empty.
    chunksize : int, default 1_000_000
        Rows read per chunk. Lower it on machines with little memory.

    Returns
    -------
    pandas.DataFrame
        Columns ``id`` (float64), ``floorspaceid`` (str),
        ``occupancystatus`` (str), ``headcount`` (float64) and
        ``collecteddate`` (str, unparsed). Rows keep their file order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``floorspace_ids`` is empty, ``chunksize`` is not positive, or
        the file lacks a required column.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw occupancy file not found: {path}")
    wanted = set(floorspace_ids)
    if not wanted:
        raise ValueError("floorspace_ids must contain at least one id")
    if chunksize <= 0:
        raise ValueError(f"chunksize must be positive, got {chunksize}")

    header = pd.read_csv(path, nrows=0)
    missing = [c for c in RAW_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {missing}")

    kept: list[pd.DataFrame] = []
    # dtype=str: nothing is converted silently while reading.
    reader = pd.read_csv(path, usecols=list(RAW_COLUMNS), dtype=str, chunksize=chunksize)
    for chunk in reader:
        chunk = chunk[chunk["floorspaceid"].isin(wanted)]
        if chunk.empty:
            continue
        chunk = chunk.assign(
            id=pd.to_numeric(chunk["id"], errors="coerce"),
            headcount=pd.to_numeric(chunk["headcount"], errors="coerce"),
        )
        kept.append(chunk)

    if not kept:
        return pd.DataFrame(
            {
                "id": pd.Series(dtype="float64"),
                "floorspaceid": pd.Series(dtype="str"),
                "occupancystatus": pd.Series(dtype="str"),
                "headcount": pd.Series(dtype="float64"),
                "collecteddate": pd.Series(dtype="str"),
            }
        )
    return pd.concat(kept, ignore_index=True)[list(RAW_COLUMNS)]


def save_processed(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a processed per-room time series to a compressed CSV.

    Compressed CSV needs no extra dependency and is readable by any tool.
    Floats are written with 10 significant figures, so a round trip is
    accurate to about 1e-10 relative.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series indexed by a tz-aware UTC ``DatetimeIndex``.
    path : str or pathlib.Path
        Destination. Gzip compression is inferred from a ``.gz`` suffix.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    ValueError
        If the index is not a tz-aware ``DatetimeIndex`` in UTC.
    """
    path = Path(path)
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or str(idx.tz) != "UTC":
        raise ValueError("df must be indexed by a tz-aware UTC DatetimeIndex")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.rename_axis(PROCESSED_INDEX)
    compression = {"method": "gzip", "compresslevel": 6} if path.suffix == ".gz" else None
    out.to_csv(path, float_format="%.10g", date_format="%Y-%m-%dT%H:%M:%SZ", compression=compression)
    return path


def load_processed(path: str | Path) -> pd.DataFrame:
    """Load a time series written by :func:`save_processed`.

    Parameters
    ----------
    path : str or pathlib.Path
        File written by :func:`save_processed`.

    Returns
    -------
    pandas.DataFrame
        Indexed by a tz-aware UTC ``DatetimeIndex`` named ``t_utc``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_csv(path, index_col=PROCESSED_INDEX)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%dT%H:%M:%SZ", utc=True)
    df.index.name = PROCESSED_INDEX
    return df
