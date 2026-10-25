"""Tests for :mod:`occvent.io` and the end-to-end :mod:`occvent.prepare` pipeline."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from helpers import ROOM_ID
from occvent.io import load_processed, read_occupancy_events, save_processed
from occvent.prepare import main

OTHER_ID = "0f3f37f8-0335-41e0-ae87-a1d677dfc7d7"  # an unmapped space in the real export

RAW_CSV = (
    '"id","deviceid","floorspaceid","occupancystatus","headcount","collecteddate",'
    '"occupancystatuschangedate","previousoccupancystatus"\n'
    f'1,"dev","{ROOM_ID}","CurrentlyOccupied",2,"2024-05-01 10:00:00+10","x","NotOccupied"\n'
    f'2,"dev","{OTHER_ID}","CurrentlyOccupied",9,"2024-05-01 10:00:10+10","x","NotOccupied"\n'
    f'3,"dev","{ROOM_ID}","CurrentlyOccupied",4,"2024-05-01 10:00:30+10","x","CurrentlyOccupied"\n'
    f'4,"dev","{ROOM_ID}","NotOccupied",0,"2024-05-01 10:01:30+10","x","RecentlyOccupied"\n'
    f'5,"dev","{ROOM_ID}","CurrentlyOccupied",oops,"2024-05-01 10:02:00+10","x","NotOccupied"\n'
    f'6,"dev","{ROOM_ID}","NotOccupied",0,"2024-05-01 10:03:00+10","x","NotOccupied"\n'
)


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text(RAW_CSV, encoding="utf-8")
    return path


def test_read_filters_rooms_and_keeps_bad_values_as_nan(raw_csv):
    df = read_occupancy_events(raw_csv, [ROOM_ID], chunksize=2)  # forces several chunks
    assert list(df.columns) == ["id", "floorspaceid", "occupancystatus", "headcount", "collecteddate"]
    assert list(df["id"]) == [1, 3, 4, 5, 6]
    assert df["headcount"].isna().sum() == 1  # "oops" -> NaN, rejected later in cleaning


def test_read_from_zip_matches_csv(raw_csv, tmp_path):
    zpath = tmp_path / "raw.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(raw_csv, arcname="raw.csv")
    pd.testing.assert_frame_equal(read_occupancy_events(zpath, [ROOM_ID]),
                                  read_occupancy_events(raw_csv, [ROOM_ID]))


def test_read_no_matching_rows_returns_empty_frame(raw_csv):
    df = read_occupancy_events(raw_csv, ["no-such-room"])
    assert df.empty
    assert "headcount" in df.columns


def test_read_errors(raw_csv, tmp_path):
    with pytest.raises(FileNotFoundError):
        read_occupancy_events(tmp_path / "missing.csv", [ROOM_ID])
    with pytest.raises(ValueError, match="at least one"):
        read_occupancy_events(raw_csv, [])
    with pytest.raises(ValueError, match="chunksize"):
        read_occupancy_events(raw_csv, [ROOM_ID], chunksize=0)
    bad = tmp_path / "bad.csv"
    bad.write_text("id,floorspaceid\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        read_occupancy_events(bad, [ROOM_ID])


def test_save_load_round_trip(tmp_path):
    idx = pd.date_range("2024-05-01", periods=4, freq="1min", tz="UTC", name="t_utc")
    df = pd.DataFrame({"headcount": [1 / 3, np.nan, 2.0, 0.0],
                       "valid": [True, False, True, True]}, index=idx)
    path = save_processed(df, tmp_path / "sub" / "x.csv.gz")
    back = load_processed(path)
    pd.testing.assert_index_equal(back.index, df.index, check_exact=True)
    np.testing.assert_allclose(back["headcount"], df["headcount"], rtol=1e-9, equal_nan=True)
    assert back["valid"].dtype == bool
    assert list(back["valid"]) == list(df["valid"])


def test_save_rejects_non_utc_index(tmp_path):
    df = pd.DataFrame({"a": [1]}, index=pd.date_range("2024-05-01", periods=1, freq="1min"))
    with pytest.raises(ValueError, match="UTC"):
        save_processed(df, tmp_path / "x.csv.gz")
    with pytest.raises(ValueError, match="UTC"):
        save_processed(df.tz_localize("Australia/Melbourne"), tmp_path / "x.csv.gz")


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_processed(tmp_path / "nope.csv.gz")


def test_prepare_end_to_end(raw_csv, tmp_path):
    out = tmp_path / "processed"
    assert main(["--raw", str(raw_csv), "--rooms", "G20", "--out", str(out)]) == 0

    grid = load_processed(out / "G20_occupancy_1min.csv.gz")
    report = json.loads((out / "G20_cleaning_report.json").read_text(encoding="utf-8"))

    # Events (UTC): 00:00:00 n=2, 00:00:30 n=4, 00:01:30 n=0, 00:03:00 n=0 ("oops" row rejected).
    np.testing.assert_allclose(grid["headcount"], [3.0, 2.0, 0.0])
    assert grid.index[0] == pd.Timestamp("2024-05-01 00:00", tz="UTC")
    assert {"night_occupied", "over_capacity", "coverage", "valid"} <= set(grid.columns)
    assert report["cleaning"]["rows_in"] == 5
    assert report["cleaning"]["rejected_bad_headcount"] == 1
    assert report["cleaning"]["rows_out"] == 4
    assert report["resampling"]["n_bins"] == 3
    assert report["room"]["room"] == "G.20"


def test_prepare_unknown_room(raw_csv, tmp_path):
    with pytest.raises(KeyError, match="valid keys"):
        main(["--raw", str(raw_csv), "--rooms", "Z99", "--out", str(tmp_path)])


def test_prepare_room_without_rows(raw_csv, tmp_path):
    with pytest.raises(ValueError, match="No rows for room G25"):
        main(["--raw", str(raw_csv), "--rooms", "G25", "--out", str(tmp_path)])
