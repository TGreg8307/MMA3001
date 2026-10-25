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
**Data-quality comparison:** G.25 (manufacturing lab) is processed by the same pipeline. Its frequent
night-time counts (26 % of night bins, a suspected sensor artefact) are flagged, which is why G.20 was
chosen for the analysis. G.25 is not analysed further.

## Key results (G.20, stated assumptions)

| Question | Answer | Where |
|---|---|---|
| Is the solver correct? | Exact solution verified analytically; Euler/RK4 orders 1.005/4.04; CO2 budget closes to 2e-14 | notebook 03 |
| Does the design rate (600 L/s) comply? | Yes, with a large margin | notebook 04 |
| Minimum constant ventilation | 211.9 L/s (NCC 8-h, 850 ppm); 443.7 L/s (1000 ppm peak) | notebook 04 |
| Demand-controlled ventilation | 12.06 L/s per person; −91 % outdoor-air conditioning energy vs the design rate | notebook 04 |
| Robust design (95 % confidence) | about 257 L/s (8-h) and 603 L/s (peak); room volume matters only for the peak | notebook 05 |
| Can a day-ahead forecast replace the sensor? | No: forecast-scheduled ventilation fails on irregular event days; sensor DCV complies | notebook 06 |

## Project status

| Stage | Status |
|---|---|
| 1. Data loading, cleaning and resampling (`occvent.io`, `occvent.cleaning`, `occvent.prepare`) | ✅ done and tested |
| 2. CO2 mass-balance model, exact solution and numerical solvers (`occvent.co2model`, `occvent.solvers`) | ✅ done and tested |
| 3. Verification and performance study (`notebooks/03_verification.ipynb`, `occvent.metrics`, `occvent.verification`) | ✅ done |
| 4. Compliance check, minimum ventilation (root finding), DCV energy comparison (`notebooks/04_ventilation_design.ipynb`, `occvent.design`) | ✅ done |
| 5. Sensitivity and uncertainty analysis (`notebooks/05_sensitivity.ipynb`, `occvent.sensitivity`) | ✅ done |
| 6. Day-ahead occupancy forecasting, ML extension (`notebooks/06_forecasting.ipynb`, `occvent.forecast`) | ✅ done |
| 7. HTML documentation, test report, written report and presentation | ✅ done |

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
│   ├── verification.py   convergence, stability, stiffness, FLOPs, timing, roofline
│   ├── design.py         compliance, Q_min root finding, DCV, energy
│   ├── sensitivity.py    scaling law, OAT/tornado, elasticities, Monte Carlo
│   └── forecast.py       day-ahead features (no leakage), baselines, models, metrics
├── tests/                pytest test suite
├── data/                 raw/ and processed/ (not committed, see data/README.md)
├── notebooks/            03_verification, 04_ventilation_design, 05_sensitivity, 06_forecasting
├── docs/                 api/ HTML documentation (pdoc), test_report.html (pytest-html), figures/
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

4. Reproduce the analysis by running the notebooks in order (`python -m jupyterlab`, then
   **Run → Run All Cells** in each). Each notebook regenerates its figures in `docs/figures/`.

| Notebook | Content | Runtime |
|---|---|---|
| `03_verification.ipynb` | solver verification, stability, convergence, performance (FLOPs, roofline, profiling) | ~1 min |
| `04_ventilation_design.ipynb` | compliance, minimum ventilation by root finding, DCV, energy | ~30 s |
| `05_sensitivity.ipynb` | scaling law, tornado, elasticities, Monte Carlo, numerical sensitivity | ~3 min |
| `06_forecasting.ipynb` | day-ahead occupancy forecasting (ML) and its ventilation consequences | ~1.5 min |

Stages 1–2 (data pipeline, CO2 model and solvers) are library code in `src/occvent/`, exercised by
the tests and by notebook 03.

## Tests and documentation

```powershell
pytest                                                      # run the test suite (120 tests)
pytest --html=docs/test_report.html --self-contained-html   # HTML test report
pdoc src/occvent -o docs/api --docformat numpy --math       # HTML API documentation
```

* **API documentation:** open `docs/api/index.html` in a browser. It is generated from the NumPy-style
  docstrings in `src/occvent/`, so it always matches the code. Re-run the `pdoc` command after any change.
* **Test report:** open `docs/test_report.html`. The tests check the code against hand calculations,
  analytical solutions (steady state, exponential decay), conservation of person-time and CO2,
  convergence orders, and leakage-free forecasting features.
* The tests need no raw data: they use small synthetic inputs.

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

## Ventilation design (stage 4)

`notebooks/04_ventilation_design.ipynb` (about 30 s) applies the verified exact solution to the whole
two-year G.20 record. Every result rests on the stated assumptions (V = 600 m3 estimated,
G = 0.0052 L/s per person, C_out = 420 ppm, well-mixed air).

* **Design occupancy:** 60 people (highest 30-minute sustained count; set by event days such as
  early-August Sundays, consistent with Open Day). Design rate 10 L/s per person gives 600 L/s.
* **Minimum constant ventilation** (root of F(Q) = metric − limit, shooting-method structure):

| Criterion | Q_min | Bisection | Secant | Secant in 1/Q | SciPy brentq |
|---|---|---|---|---|---|
| NCC 8-h mean ≤ 850 ppm | 211.9 L/s (35 % of design) | 17 solves | 9 | **6** | 211.86 ✓ |
| Peak ≤ 1000 ppm | 443.7 L/s (74 % of design) | 17 solves | 12 | **7** | 443.68 ✓ |

* **Demand-controlled ventilation** Q = max(50 L/s, q_p N): q_p = 12.06 L/s per person (theory 12.09),
  meeting both criteria, also with the fan capped at 600 L/s.
* **Annual outdoor-air conditioning energy** (ΔT = 7 K assumed; percentages are independent of ΔT):
  fixed design 44.4 MWh, scheduled 06–22 h 23.1 MWh (−48 %), DCV 3.9 MWh (**−91 %**).

## Sensitivity and uncertainty (stage 5)

`notebooks/05_sensitivity.ipynb` (about 3 minutes; section G3 also re-reads the raw zip if present)
tests how the Stage 4 answers depend on the uncertain inputs: room volume V (300–1200 m3),
CO2 generation G (0.0040–0.0060 L/s), outdoor CO2 (400–460 ppm) and a people-counter
factor k (true occupancy = k × counted, 0.8–1.2).

* **Exact scaling law.** The model is linear, so every metric is
  `C_out + (k G / V) M(Q/V)` for one unit-room curve M(a). G and k act only as the product kG.
  Q_min for any inputs follows from interpolating the tabulated curve (checked against full
  root finding: max error 0.014 %), so 20 000 Monte Carlo draws take milliseconds.
* **One-at-a-time (tornado):** Q_min for the NCC 8-h criterion is driven by kG (about ±20 %) and C_out;
  the room volume barely matters (< 3 %). For the 1000 ppm peak, V matters as much as kG (about ±22 %).
  The DCV rate q_p depends on kG and C_out only.
* **Elasticities** from the full model agree with the theory from the scaling law.
* **Monte Carlo (20 000 draws):**

| Criterion | Nominal Q_min | 5th–95th percentile | P(nominal is enough) | P(600 L/s design fails) |
|---|---|---|---|---|
| NCC 8-h mean ≤ 850 ppm | 211.9 L/s | 168–257 L/s | 54 % | 0 % |
| Peak ≤ 1000 ppm | 443.7 L/s | 301–603 L/s | 54 % | 5.3 % |

* **Numerical sensitivity:** RK4 changes Q_min by < 0.01 L/s and explicit Euler by up to 1 % (peak).
  Hold limits make no difference. Coarse occupancy bins under-size the peak design (−25 % at 60 min),
  so 1-minute data are kept.

## Day-ahead occupancy forecasting (stage 6, ML extension)

`notebooks/06_forecasting.ipynb` (about 1.5 minutes) asks whether tomorrow's 15-minute occupancy can be
forecast well enough to schedule ventilation without a live sensor. Forecasts are issued at local midnight
and use only the calendar (time of day, weekday, Victorian public holidays) and occupancy from earlier days;
a unit test confirms no information from the target day leaks into the features.

* **Split (time-ordered):** training 31 Mar 2024 – 30 Jun 2025, validation Jul–Dec 2025, test
  1 Jan – 22 Apr 2026 (used once). Tuning by `TimeSeriesSplit` (5 folds, 1-day gap).
* **Models (Week 5):** mean (`DummyRegressor`), seasonal naive, 4-week mean, linear regression, decision
  tree (depth 6, leaf ≥ 200 by CV), neural network (3 seeds). Selected: **linear regression**.

| Test period (Jan–Apr 2026) | MAE | RMSE | R² | Busy-slot bias (≥ 5 people) |
|---|---|---|---|---|
| Seasonal naive (same time last week) | 0.90 | 2.73 | −0.31 | −5.0 |
| 4-week mean (2 h smoothed) | 0.75 | 2.11 | 0.22 | −5.5 |
| Linear regression (selected) | 0.76 | **2.02** | **0.29** | −5.7 |

* **Engineering test:** ventilation scheduled from the forecast, `Q = max(50, s q_p N̂)`, exceeds the
  NCC 8-h limit on 12 of 112 test days (s = 1) and still on 2 days even with s = 8 (at twice the airflow),
  because unforeseen events (e.g. up to 59 people on the evening of 19 Feb 2026) are invisible to the
  forecast. Sensor-driven DCV complies on every day. **Conclusion:** the forecast is useful for planning,
  but not as a replacement for a live occupancy or CO2 sensor.

## AI use

AI assistance (Claude) was used in this project. Its use is logged
throughout development and reported, with verification evidence, in the
AI-use reflection section of the project report.

## Licence

Code: MIT (see [LICENSE](LICENSE)). Data: not included, owned by Monash
University.
