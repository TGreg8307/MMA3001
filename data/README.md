# Data

The data is **not** stored in this repository. The raw export is about
1.6 GB, and it belongs to Monash University (Smart Infrastructure / MMA3001
teaching team).

## Raw data (`data/raw/`)

Put the file supplied for the unit here, unchanged:

```
data/raw/5occupancySensor_MayToDec2024_9MRows.zip
```

The pipeline reads the `.zip` directly, so there is no need to extract the
1.6 GB CSV. Despite the filename, the export covers **24 Nov 2023 to
22 Apr 2026** and has **9,091,732 rows** across 5 floor spaces from one
gateway device.

| Column | Meaning | Used |
|---|---|---|
| `id` | Row id, increasing in the order rows were written | yes (tie-break) |
| `deviceid` | Gateway id (one value for every row) | no |
| `floorspaceid` | Monitored space (UUID) | yes |
| `occupancystatus` | `CurrentlyOccupied` / `RecentlyOccupied` / `NotOccupied` | yes (validation) |
| `headcount` | People counted (integer, ≥ 0) | yes |
| `collecteddate` | Event time with UTC offset, e.g. `2024-05-01 09:00:00+10` | yes |
| `occupancystatuschangedate` | Time of the last status change | no |
| `previousoccupancystatus` | Status before this event | no |

Room mapping (from `Sensor ID and Locations.xlsx`, sheet "OCC", which is
titled "Incorrect Data", so the mapping is an assumption):

| Key | Room | `floorspaceid` |
|---|---|---|
| G20 | Building 60, G.20 Informal Breakout | `cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b` |
| G25 | Building 60, G.25 Manufacturing lab | `227f1068-9e41-469f-bef0-275f27a0bdba` |

## Processed data (`data/processed/`)

Created by `python -m occvent.prepare`. For each room:

* `<ROOM>_occupancy_1min.csv.gz`: one row per UTC minute.
  * `t_utc`: start of the minute (ISO-8601, UTC).
  * `headcount`: time-averaged people in the minute.
  * `coverage`: observed fraction of the minute.
  * `valid`: True if the minute is fully observed.
  * `night_occupied`: count > 0 between 01:00 and 05:00 local time.
  * `over_capacity`: count above a set capacity (off by default).
* `<ROOM>_cleaning_report.json`: counts of rejected, duplicate and
  out-of-order rows, hold truncations, and quality statistics.
