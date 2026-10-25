"""Command-line entry point: turn the raw export into clean per-room series.

Usage (from the repository root)::

    python -m occvent.prepare --raw data/raw/5occupancySensor_MayToDec2024_9MRows.zip

For each room this writes to ``data/processed/``:

* ``<ROOM>_occupancy_<step>.csv.gz``: the regular time series (see
  :func:`occvent.cleaning.resample_occupancy` and
  :func:`occvent.cleaning.add_quality_flags`).
* ``<ROOM>_cleaning_report.json``: every count needed to audit the cleaning.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from .cleaning import ResampleSettings, add_quality_flags, clean_events, resample_occupancy
from .config import LOCAL_TIMEZONE, ROOMS, get_room
from .io import read_occupancy_events, save_processed

log = logging.getLogger("occvent.prepare")


def prepare_rooms(
    raw_path: str | Path,
    room_keys: Sequence[str],
    out_dir: str | Path,
    settings: ResampleSettings | None = None,
    night_hours: tuple[int, int] = (1, 5),
) -> dict[str, dict]:
    """Run the full cleaning pipeline for several rooms in one pass of the file.

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Raw export (``.csv`` or ``.zip``).
    room_keys : sequence of str
        Keys of :data:`occvent.config.ROOMS` to process, e.g. ``["G20", "G25"]``.
    out_dir : str or pathlib.Path
        Output folder. Created if it does not exist.
    settings : ResampleSettings, optional
        Resampling settings. Defaults to :class:`ResampleSettings()`.
    night_hours : tuple of int, default (1, 5)
        Local night window passed to :func:`occvent.cleaning.add_quality_flags`.

    Returns
    -------
    dict
        ``{room_key: report}``, where each report is also saved as JSON.

    Raises
    ------
    KeyError
        If a room key is unknown.
    ValueError
        If a requested room has no rows in the raw file.
    """
    settings = settings or ResampleSettings()
    rooms = [get_room(k) for k in room_keys]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    raw = read_occupancy_events(raw_path, [r.floorspace_id for r in rooms])
    log.info("Read %d rows for %d room(s) in %.1f s", len(raw), len(rooms), time.perf_counter() - t0)

    reports: dict[str, dict] = {}
    for room in rooms:
        sub = raw[raw["floorspaceid"] == room.floorspace_id]
        if sub.empty:
            raise ValueError(f"No rows for room {room.key} ({room.floorspace_id}) in {raw_path}")
        events, clean_rep = clean_events(sub)
        grid, resample_rep = resample_occupancy(events, settings)
        grid = add_quality_flags(grid, night_hours=night_hours)

        valid = grid["valid"]
        night_bins = grid.index.tz_convert(LOCAL_TIMEZONE).hour
        in_night = (night_bins >= night_hours[0]) & (night_bins < night_hours[1])
        n_night_valid = int((valid & in_night).sum())
        report = {
            "room": {"key": room.key, "building": room.building, "room": room.room,
                     "floorspace_id": room.floorspace_id, "description": room.description},
            "raw_file": Path(raw_path).name,
            "cleaning": clean_rep,
            "resampling": resample_rep,
            "quality": {
                "night_hours_local": list(night_hours),
                "fraction_of_valid_night_bins_occupied": (
                    float((grid["night_occupied"] & valid).sum() / n_night_valid) if n_night_valid else None
                ),
                "max_headcount_bin_mean": float(grid["headcount"].max()),
                "mean_headcount_valid_bins": float(grid.loc[valid, "headcount"].mean()),
            },
        }

        step_tag = settings.step.replace(" ", "")
        series_path = save_processed(grid, out_dir / f"{room.key}_occupancy_{step_tag}.csv.gz")
        report_path = out_dir / f"{room.key}_cleaning_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("%s: %d events -> %d bins (%.1f%% valid) -> %s",
                 room.key, clean_rep["rows_out"], resample_rep["n_bins"],
                 100 * resample_rep["fraction_valid"], series_path.name)
        reports[room.key] = report
    return reports


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line argument parser."""
    defaults = ResampleSettings()
    p = argparse.ArgumentParser(
        prog="python -m occvent.prepare",
        description="Clean the raw occupancy export and resample it to a regular grid.",
    )
    p.add_argument("--raw", required=True, type=Path, help="raw export (.csv or .zip)")
    p.add_argument("--rooms", nargs="+", default=sorted(ROOMS), help="room keys (default: all)")
    p.add_argument("--out", type=Path, default=Path("data/processed"), help="output folder")
    p.add_argument("--step", default=defaults.step, help="grid step, e.g. 1min")
    p.add_argument("--max-occupied-hold", default=defaults.max_occupied_hold)
    p.add_argument("--max-vacant-hold", default=defaults.max_vacant_hold)
    p.add_argument("--min-coverage", type=float, default=defaults.min_coverage)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = ResampleSettings(
        step=args.step,
        max_occupied_hold=args.max_occupied_hold,
        max_vacant_hold=args.max_vacant_hold,
        min_coverage=args.min_coverage,
    )
    prepare_rooms(args.raw, args.rooms, args.out, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
