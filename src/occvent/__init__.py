"""occvent: occupancy-driven ventilation assessment.

MMA3001 (Numerical Methods and Machine Learning) individual project.

This package turns event-driven people-counting data from the Monash Smart
Infrastructure sensors into a clean, regular occupancy signal N(t). Later
stages use N(t) to drive a well-mixed CO2 mass-balance model of a room,
which is used to assess and optimise its ventilation.

Modules
-------
config
    Constants, the room registry and the time zone.
io
    Chunked reading of the raw export; saving and loading processed series.
cleaning
    Event validation, de-duplication, exact time-weighted resampling and
    data-quality flags.
prepare
    Command-line pipeline: ``python -m occvent.prepare --raw <file>``.
"""

__version__ = "0.1.0"
