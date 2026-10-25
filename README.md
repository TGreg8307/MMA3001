# Occupancy-Driven Ventilation Assessment

**MMA3001 Numerical Methods and Machine Learning, individual project (2026)**
Author: Tom Gregory

## Engineering problem

Mechanical ventilation is one of the largest energy uses in non-residential
buildings. It must also keep indoor CO2 within limits: the NCC verification
limit is 850 ppm averaged over 8 hours. Ventilation is usually sized for a
*design* occupancy, and most rooms have no CO2 sensor to show how they
actually perform.

This project uses **measured people-counts only** to:

1. estimate a room's indoor CO2 with a well-mixed CO2 mass-balance ODE, a
   "virtual CO2 sensor";
2. check compliance at the AS 1668.2 design ventilation rate;
3. find the minimum ventilation rate that meets the CO2 criterion;
4. quantify the outdoor-air and energy savings of demand-controlled
   ventilation compared with a fixed rate.

The measured occupancy is piecewise constant between sensor events, so the
ODE has an exact solution. It is used to verify the numerical solvers
(explicit and implicit Euler, RK4, SciPy `solve_ivp`).

**Case study:** Monash Clayton, Building 60, room G.20 (Informal Breakout).
**Robustness case:** G.25 (manufacturing lab).

## Project status

| Stage | Status |
|---|---|
| 1. Data loading, cleaning and resampling (`occvent.io`, `occvent.cleaning`, `occvent.prepare`) | ✅ done and tested |
| 2. CO2 mass-balance model, exact solution and numerical solvers (`occvent.co2model`, `occvent.solvers`) | ✅ done and tested |
| 3. Verification and performance study (`notebooks/03_verification.ipynb`, `occvent.metrics`, `occvent.verification`) | ✅ done |
| 4. Compliance check, ventilation optimisation, DCV energy comparison | ⏳ |
| 5. Sensitivity analysis | ⏳ |
| 6. Occupancy forecasting (ML extension) | ⏳ |

## Repository layout

```
├── src/occvent/          Python package (all project code)
│   ├── config.py         constants, room registry, time zone
│   ├── io.py             chunked raw reader; save/load processed series
│   ├── cleaning.py       validation, de-duplication, exact resampling, quality flags
│   ├── prepare.py        command-line data pipeline
│   ├── co2model.py       CO2 mass-balance ODE, exact solution, mass balance
│   ├── solvers.py        explicit/implicit Euler, RK4, SciPy benchmark
│   ├── metrics.py        8-hour mean, peak, time above limits
│   └── verification.py   convergence, stability, stiffness, FLOPs, timing, roofline
├── tests/                pytest test suite
├── data/                 raw/ and processed/ (not committed, see data/README.md)
├── notebooks/            analysis notebooks (03_verification.ipynb)
├── docs/                 figures/ for the report; HTML API documentation (pdoc)
├── pyproject.toml        package metadata and pytest settings
└── requirements.txt      pinned, verified dependency versions
```

## Setup (Windows PowerShell)

Requires Python 3.11 or newer.

```powershell
git clone <repository-url>
cd <repository-folder>
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

(macOS/Linux: `python3 -m venv .venv` and `source .venv/bin/activate`.)

## Usage

1. Copy the raw export into `data/raw/` (see [data/README.md](data/README.md)).
2. Build the clean per-room series (about 1 minute; reads the `.zip` directly):

```powershell
python -m occvent.prepare --raw data/raw/5occupancySensor_MayToDec2024_9MRows.zip
```

Options: `--rooms G20 G25`, `--step 1min`, `--max-occupied-hold 2.5h`,
`--max-vacant-hold 96h`, `--min-coverage 1.0`, `--out data/processed`.

3. Load a series in Python or a notebook:

```python
from occvent.io import load_processed
g20 = load_processed("data/processed/G20_occupancy_1min.csv.gz")
```

## Tests and documentation

```powershell
pytest                                                  # run the test suite
pytest --html=docs/test_report.html --self-contained-html   # HTML test report
pdoc src/occvent -o docs/api --docformat numpy --math   # HTML API docs
```

## Data cleaning method (stage 1)

The sensor writes a row only when the count changes, so the raw data is an
irregular event log. The pipeline:

1. **Reads** only the needed columns and rooms, in 1-million-row chunks.
2. **Validates** each row: parseable timestamp, non-negative integer count,
   known status, present id, and status consistent with count. Each
   rejected row is counted in a JSON report.
3. **Converts to UTC.** Raw offsets switch between +10 and +11 with daylight
   saving. Rows are sorted by `(time, id)`; equal timestamps keep the last
   row written.
4. **Resamples exactly.** Counts are held constant between events (a
   zero-order hold), and each 1-minute bin gets the exact time-average of
   that step function. This conserves person-time, and therefore CO2
   generation. Integer-second arithmetic makes it exact, and the cumulative
   sums are vectorised (about 0.25 s per room for 1.1 M bins).
5. **Limits holds.** A non-zero count is trusted for at most 2.5 h, a zero
   count for at most 96 h. Beyond that the data is treated as unobserved
   (`valid = False`). These limits come from the data audit: in G.20,
   occupied holds never exceed 2.0 h and vacant holds reach 87 h (long
   weekends).
6. **Flags** night-time occupancy (01:00 to 05:00 local) without removing it.

Results on the real export:

| | G.20 | G.25 |
|---|---|---|
| Events kept / read | 1,894,867 / 1,894,871 | 1,500,391 / 1,500,394 |
| Period | 3 Mar 2024 to 22 Apr 2026 | 3 Mar 2024 to 22 Apr 2026 |
| Valid 1-min bins | 99.9998 % | 99.86 % |
| Occupied holds cut at 2.5 h (likely stuck counts) | 0 | 11 (25.2 h) |
| Night bins occupied | 3.2 % | 25.8 % (suspected sensor artefact) |
| Mean occupancy | 0.89 people | 0.66 people |

## CO2 model and solvers (stage 2)

Well-mixed single-zone mass balance (`occvent.co2model`):

    V dC/dt = 1e6 G N(t) + Q (C_out - C)      ->   dC/dt = a - k C,  k = Q/V

| Symbol | Meaning | Default |
|---|---|---|
| V | room air volume [m3] | G.20: **600 m3, estimated** (no dimensions supplied; varied in sensitivity analysis) |
| Q | outdoor-air flow [L/s] | input (constant or time-varying) |
| G | CO2 generation [L/s per person] | 0.0052 |
| C_out | outdoor CO2 [ppm] | 420 |

Because N and Q are constant within each 1-minute step, the exact step
solution `C_{i+1} = e^{-kh} C_i + a h (1 - e^{-kh})/(kh)` is applied step by
step. It has no discretisation error, so it is the reference for verifying
the numerical solvers (`occvent.solvers`).

Results on the full G.20 record (1,123,729 one-minute steps, V = 600 m3,
Q = 300 L/s, so k h = 0.03):

| Method | Runtime | Max error vs exact |
|---|---|---|
| Exact, `lfilter` (compiled) | 0.06 s | reference |
| Exact, Python loop | 0.24 s | 0 (identical) |
| Explicit Euler (h = 60 s) | 0.33 s | 5.2 ppm |
| Implicit Euler (h = 60 s) | 0.48 s | 5.1 ppm |
| RK4 (h = 60 s) | 0.68 s | 2.4e-6 ppm |
| SciPy RK45 (rtol 1e-8), **one week only** | 4.2 s | 3.6e-3 ppm |

Observed orders of convergence (test problem with k dt = 0.5, step halved
4 times): explicit Euler 1.02, implicit Euler 0.98, RK4 4.04, matching
theory (1, 1, 4). The CO2 budget closes to a relative residual of 2e-14.

## Verification and performance (stage 3)

`notebooks/03_verification.ipynb` (about 1 minute to run; needs the processed data and the raw zip)
checks the solvers against the exact solution and measures their cost. Figures are saved to
`docs/figures/`. Main results:

| Check | Result |
|---|---|
| Convergence orders (stress problem, k dt = 0.5) | explicit Euler 1.005, implicit Euler 0.995, RK4 4.04 |
| Engineering output (max 8-h mean, real week) | error < 0.1 ppm for every solver at the 1-min step |
| Input resolution | 60-min occupancy bins lower the weekly peak by 38 ppm; 8-h mean changes by < 3 ppm |
| Stability | Euler fails for k h > 2, RK4 for k h > 2.785, implicit Euler never |
| Stiffness | one eigenvalue (-Q/V): not stiff; implicit needed only for coarse data in small, well-ventilated rooms |
| Exact vs trapezoidal 8-h mean | trapezoid exact to 1e-5 ppm at 1-min sampling; 0.25–1.7 ppm error at 16–60 min sampling (corners in C) |
| Zero-order hold vs linear interpolation | linear interpolation overstates person-hours by 5.2 % on the busiest day |
| Numerical vs parameter error | Euler error 0.09 ppm vs 50–60 ppm for ±20 % in Q or G |
| Performance (1.12 M steps) | exact `lfilter` 0.04–0.05 s (3–6× faster than the Python loop, machine-dependent); loop solvers interpreter-bound, far below the measured roofline |

## AI use

AI assistance (Claude) was used in this project. Its use is logged
throughout development and reported, with verification evidence, in the
AI-use reflection section of the project report.

## Licence

Code: MIT (see [LICENSE](LICENSE)). Data: not included, owned by Monash
University.
