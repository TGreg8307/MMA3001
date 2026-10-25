"""Project-wide constants and the registry of rooms studied.

All site-specific values are kept here so that no other module hard-codes
sensor identifiers, column names or time zones.
"""

from __future__ import annotations

from dataclasses import dataclass

#: IANA time zone of the Monash Clayton campus. Raw timestamps carry a UTC
#: offset of +10 (AEST) or +11 (AEDT); this zone gives the local calendar
#: time used for night-time checks and calendar features.
LOCAL_TIMEZONE: str = "Australia/Melbourne"

#: Raw CSV columns the pipeline reads. The other columns in the export
#: (``deviceid``, ``occupancystatuschangedate``, ``previousoccupancystatus``)
#: are not needed: ``deviceid`` is one gateway for every row, and the status
#: history can be rebuilt from the event sequence itself.
RAW_COLUMNS: tuple[str, ...] = (
    "id",
    "floorspaceid",
    "occupancystatus",
    "headcount",
    "collecteddate",
)

#: Allowed values of ``occupancystatus``.
VALID_STATUSES: frozenset[str] = frozenset(
    {"CurrentlyOccupied", "RecentlyOccupied", "NotOccupied"}
)

#: The only status that may carry a non-zero headcount.
OCCUPIED_STATUS: str = "CurrentlyOccupied"

#: Format of ``collecteddate`` in the raw export, e.g. ``2024-03-05 12:32:48+11``.
TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M:%S%z"


@dataclass(frozen=True)
class Room:
    """Identity of one monitored space.

    Attributes
    ----------
    key : str
        Short identifier used on the command line and in file names.
    floorspace_id : str
        Value of ``floorspaceid`` in the raw occupancy export.
    building : int
        Monash Clayton building number.
    room : str
        Room number as printed in ``Sensor ID and Locations.xlsx``.
    description : str
        Human-readable description and role in the study.
    """

    key: str
    floorspace_id: str
    building: int
    room: str
    description: str


#: Rooms analysed in this project. The mapping of ``floorspace_id`` to room
#: comes from the "OCC" sheet of ``Sensor ID and Locations.xlsx``. That sheet
#: is titled "Incorrect Data", so the mapping is an assumption stated in the
#: report.
ROOMS: dict[str, Room] = {
    "G20": Room(
        key="G20",
        floorspace_id="cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b",
        building=60,
        room="G.20",
        description="Informal Breakout - primary case study",
    ),
    "G25": Room(
        key="G25",
        floorspace_id="227f1068-9e41-469f-bef0-275f27a0bdba",
        building=60,
        room="G.25",
        description="Manufacturing lab (MSM) - robustness case",
    ),
}


#: Default outdoor (supply-air) CO2 concentration [ppm]. Global background
#: CO2 was about 420-425 ppm in 2024-2026. The NCC note behind its 850 ppm
#: limit assumes 400 ppm ambient. Varied in the sensitivity analysis.
DEFAULT_OUTDOOR_CO2_PPM: float = 420.0

#: Default CO2 generation per occupant [L/s per person]. Typical adult value
#: for sedentary or light office activity (about 1.2 met), as used in
#: ASHRAE 62.1 / ASTM D6245. The literature range is about 0.004-0.006 L/s.
#: Varied in the sensitivity analysis.
DEFAULT_CO2_GENERATION_LPS: float = 0.0052

#: ESTIMATED internal air volume of each room [m3]. No dimensions were
#: supplied. G.20 is taken as about 200 m2 of floor area x 3.0 m ceiling.
#: This is an assumption: results are reported with a sensitivity range
#: (about 300-1200 m3) and should be updated if real dimensions are obtained.
ESTIMATED_VOLUME_M3: dict[str, float] = {
    "G20": 600.0,
}


def get_room(key: str) -> Room:
    """Return the :class:`Room` registered under ``key``.

    Parameters
    ----------
    key : str
        Room key, e.g. ``"G20"``. Matching ignores case.

    Returns
    -------
    Room
        The registered room.

    Raises
    ------
    KeyError
        If ``key`` is not a registered room. The message lists the valid keys.
    """
    try:
        return ROOMS[key.upper()]
    except KeyError:
        valid = ", ".join(sorted(ROOMS))
        raise KeyError(f"Unknown room {key!r}; valid keys are: {valid}") from None
