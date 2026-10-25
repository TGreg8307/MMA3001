"""Shared helpers for the test suite.

The tests use small synthetic event tables whose correct cleaned and
resampled output can be worked out by hand.
"""

from __future__ import annotations

import pandas as pd

ROOM_ID = "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b"  # G.20


def make_raw(rows: list[tuple]) -> pd.DataFrame:
    """Build a raw-format DataFrame from ``(id, status, headcount, collecteddate)`` tuples."""
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "floorspaceid": ROOM_ID,
            "occupancystatus": [r[1] for r in rows],
            "headcount": [r[2] for r in rows],
            "collecteddate": [r[3] for r in rows],
        }
    )


def make_events(pairs: list[tuple[str, int]]) -> pd.DataFrame:
    """Build a cleaned-format event table from ``(utc_iso_time, headcount)`` pairs."""
    return pd.DataFrame(
        {
            "t": pd.to_datetime([p[0] for p in pairs], utc=True),
            "headcount": [p[1] for p in pairs],
        }
    )
