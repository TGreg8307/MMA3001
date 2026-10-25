"""Tests for :func:`occvent.cleaning.clean_events` and :func:`add_quality_flags`."""

from __future__ import annotations

import pandas as pd
import pytest

from helpers import make_raw
from occvent.cleaning import add_quality_flags, clean_events


def test_valid_rows_pass_and_are_converted_to_utc():
    raw = make_raw([
        (1, "CurrentlyOccupied", 3, "2024-05-01 09:00:00+10"),
        (2, "NotOccupied", 0, "2024-05-01 09:05:00+10"),
    ])
    events, rep = clean_events(raw)
    assert rep["rows_out"] == 2
    assert list(events["headcount"]) == [3, 0]
    assert events["t"].iloc[0] == pd.Timestamp("2024-04-30 23:00:00", tz="UTC")
    assert str(events["t"].dt.tz) == "UTC"


def test_daylight_saving_offsets_give_correct_order():
    # DST ended 2024-04-07 03:00 AEDT (+11) -> 02:00 AEST (+10).
    # 02:30+11 is 15:30Z on 6 Apr; 02:10+10 is 16:10Z, so it comes later
    # even though its local clock time is earlier.
    raw = make_raw([
        (2, "CurrentlyOccupied", 1, "2024-04-07 02:10:00+10"),
        (1, "CurrentlyOccupied", 2, "2024-04-07 02:30:00+11"),
    ])
    events, rep = clean_events(raw)
    assert list(events["headcount"]) == [2, 1]
    assert events["t"].is_monotonic_increasing
    assert rep["rows_out_of_order_in_file"] == 1


def test_each_invalid_row_is_rejected_and_counted_once():
    raw = make_raw([
        (1, "CurrentlyOccupied", 2, "2024-05-01 09:00:00+10"),   # valid
        (2, "CurrentlyOccupied", 2, "not-a-date"),               # bad timestamp
        (3, "CurrentlyOccupied", -1, "2024-05-01 09:01:00+10"),  # negative count
        (4, "CurrentlyOccupied", 1.5, "2024-05-01 09:02:00+10"), # non-integer count
        (5, "Occupied??", 2, "2024-05-01 09:03:00+10"),          # bad status
        (None, "CurrentlyOccupied", 2, "2024-05-01 09:04:00+10"),  # missing id
        (7, "NotOccupied", 4, "2024-05-01 09:05:00+10"),         # status/count mismatch
        (8, "CurrentlyOccupied", 0, "2024-05-01 09:06:00+10"),   # status/count mismatch
        (9, "CurrentlyOccupied", None, "bad"),                   # fails timestamp first
    ])
    events, rep = clean_events(raw)
    assert rep["rows_in"] == 9
    assert rep["rejected_bad_timestamp"] == 2
    assert rep["rejected_bad_headcount"] == 2
    assert rep["rejected_bad_status"] == 1
    assert rep["rejected_missing_id"] == 1
    assert rep["rejected_status_count_mismatch"] == 2
    assert rep["rows_out"] == 1
    assert sum(v for k, v in rep.items() if k.startswith("rejected_")) + rep["rows_out"] == rep["rows_in"]


def test_recently_occupied_with_zero_count_is_valid():
    raw = make_raw([(1, "RecentlyOccupied", 0, "2024-05-01 09:00:00+10")])
    _, rep = clean_events(raw)
    assert rep["rows_out"] == 1


def test_duplicate_timestamps_keep_highest_id():
    raw = make_raw([
        (10, "CurrentlyOccupied", 5, "2024-05-01 09:00:00+10"),
        (9, "CurrentlyOccupied", 4, "2024-05-01 09:00:00+10"),
        (11, "CurrentlyOccupied", 6, "2024-05-01 09:00:00+10"),
        (12, "NotOccupied", 0, "2024-05-01 09:01:00+10"),
    ])
    events, rep = clean_events(raw)
    assert rep["duplicates_removed"] == 2
    assert list(events["headcount"]) == [6, 0]
    assert events["t"].is_unique


def test_multiple_rooms_rejected():
    raw = make_raw([(1, "NotOccupied", 0, "2024-05-01 09:00:00+10")] * 2)
    raw.loc[1, "floorspaceid"] = "another-room"
    with pytest.raises(ValueError, match="one floorspaceid"):
        clean_events(raw)


def test_missing_column_rejected():
    raw = make_raw([(1, "NotOccupied", 0, "2024-05-01 09:00:00+10")]).drop(columns="headcount")
    with pytest.raises(ValueError, match="headcount"):
        clean_events(raw)


def test_all_rows_invalid_gives_empty_events():
    raw = make_raw([(1, "NotOccupied", 0, "garbage")])
    events, rep = clean_events(raw)
    assert events.empty
    assert rep["rows_out"] == 0
    assert rep["first_event_utc"] is None


def test_night_flag_uses_local_time():
    # 15:30Z = 01:30 AEST (+10) on 1 May, which is inside night hours (1, 5).
    # 06:00Z = 16:00 AEST, which is daytime.
    idx = pd.DatetimeIndex(["2024-05-01 15:30", "2024-05-01 06:00", "2024-05-01 15:31"], tz="UTC")
    grid = pd.DataFrame({"headcount": [2.0, 3.0, 0.0]}, index=idx)
    out = add_quality_flags(grid, night_hours=(1, 5))
    assert list(out["night_occupied"]) == [True, False, False]
    assert not out["over_capacity"].any()
    assert "night_occupied" not in grid.columns  # input is not modified


def test_capacity_flag_and_nan_handling():
    idx = pd.date_range("2024-05-01 06:00", periods=3, freq="1min", tz="UTC")
    grid = pd.DataFrame({"headcount": [10.0, float("nan"), 50.0]}, index=idx)
    out = add_quality_flags(grid, capacity=40)
    assert list(out["over_capacity"]) == [False, False, True]


def test_flag_argument_validation():
    idx = pd.date_range("2024-05-01", periods=1, freq="1min", tz="UTC")
    grid = pd.DataFrame({"headcount": [1.0]}, index=idx)
    with pytest.raises(ValueError):
        add_quality_flags(grid, night_hours=(5, 1))
    with pytest.raises(ValueError):
        add_quality_flags(grid, capacity=0)
    with pytest.raises(ValueError):
        add_quality_flags(grid.tz_localize(None))
