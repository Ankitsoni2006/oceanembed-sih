# OceanEmbed — SIH26066
### Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations

**Status: research prototype (SIH 2026 MVP).** OceanEmbed reconstructs a 15-depth (0–1000 m) ocean
temperature profile for the North Indian Ocean from seven satellite-derived surface fields, and
evaluates that reconstruction against an independent, offline set of September 2020 ARGO float
observations. It is not an operational, real-time or forecasting service: it runs on a processed
archive of daily fields for 1 January – 30 September 2020.

---

## What the system does

```
RECONSTRUCTION MODE                               ARGO VALIDATION MODE (evaluation only)
───────────────────                               ───────────────────────────────────────
7 surface fields + 7 validity masks               Authentic Sep 2020 ARGO profile
  (SST, SSS, SSH, U/V current, U/V 10 m wind)       (WMO id, time, lat/lon, P/T cast)
        │  [14 × 101 × 241], one day                      │
        ▼                                                 ▼
Train-only standard scaler (Jan–Jul 2020)         Interpolate cast onto the 15 depths
        │                                           (no extrapolation; gaps stay missing)
        ▼                                                 │
OceanEmbedNetV3_Decoder (frozen, 1,275,934 params)  Same day + nearest 0.25° grid cell
        │                                                 │
        ▼                                                 ▼
15-depth temperature field [15 × 101 × 241]  ───►  Surface-only reconstruction at that cell
        │                                                 │
        ▼                                                 ▼
Profile at the requested cell + MLD,              Observed vs predicted, per-depth error,
thermocline depth, OHC300                         RMSE / MAE / Bias / Pearson r
```

ARGO temperatures are **never** a model input. They are used only after inference, to score the
surface-only reconstruction.

---

## Model and data (verified against code and artifacts)

| Item | Value |
|:--|:--|
| Architecture | `OceanEmbedNetV3_Decoder` (U-Net encoder, 128-channel bottleneck, parallel 1×1 multi-depth head) — `src/models/oceanembed_v3_decoder.py` |
| Trainable parameters | 1,275,934 |
| Checkpoint | `checkpoints/phase5/oceanembed_v3_decoder.pt` (5,148,823 bytes) |
| Inputs | 14 channels = 7 physical surface variables + 7 validity masks |
| Outputs | 15 depths: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m |
| Grid | 5°N–30°N × 45°E–105°E at 0.25° (101 × 241) |
| Processed dates | 2020-01-01 → 2020-09-30 (274 daily fields) |
| Split | Train Jan–Jul 2020 (213 d) · Validation Aug 2020 (31 d) · Held-out test Sep 2020 (30 d) |
| Scaler | `configs/scaler_params_experiment_2020.json`, fitted on the training period only (see `..._metadata.json`) |
| Supervised target | GLORYS12V1 reanalysis, regridded to the 15 depths — a **reanalysis reference, not observational ground truth** |

---

## Evaluation results

### GLORYS reanalysis reference — held-out September 2020 (offline benchmark)
Source: `reports/phase5/final_model_comparison.json`

| Model | Params | RMSE (°C) | MAE (°C) | Bias (°C) |
|:--|--:|--:|--:|--:|
| Static climatology | 0 | 2.9976 | 2.4705 | −2.1353 |
| Pointwise MLP | 6,095 | 1.1177 | 0.7569 | +0.0547 |
| Simple CNN | 46,031 | 1.0418 | 0.7100 | −0.0052 |
| Original OceanEmbed (phase 4) | 1,342,928 | 1.8142 | 1.3229 | −0.2972 |
| **OceanEmbed v3** | **1,275,934** | **0.8601** | **0.5780** | **+0.0221** |

### ARGO in-situ observational evaluation — September 2020
36 authentic profiles, 28 unique WMO floats, 497 profile-depth observations, from three GDAC
snapshot files (2020-09-01, 2020-09-15, 2020-09-25). The v3 row is recomputed by the running backend
(`GET /argo/summary`); the baseline rows come from the offline experiment
(`reports/argo2020/argo2020_metrics.json`, `reports/phase5/final_model_comparison.json`).

| Model | RMSE (°C) | MAE (°C) | Bias (°C) |
|:--|--:|--:|--:|
| Static climatology | 3.0363 | 2.5590 | −2.2871 |
| Pointwise MLP | 1.1602 | 0.6560 | +0.1670 |
| Simple CNN | 0.9526 | 0.5874 | +0.0757 |
| **OceanEmbed v3** | **0.7973** | **0.4820** | **+0.0507** |

Notes:
* ARGO is an independent **observational check**, not a proof of statistical independence from
  GLORYS (GLORYS assimilates in-situ data).
* One of the 36 profiles (WMO 6902948) falls on a grid cell with no valid SST and only 3/7 surface
  channels. It is kept in the aggregate and flagged explicitly by the API and UI.
* The largest errors are in the 75–150 m thermocline band (≈1.5 °C RMSE at 100 m).

---

## Running the MVP

### Backend (FastAPI, port 8000)
```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Or double-click `start_backend.bat`. API docs: `http://localhost:8000/docs`.

### Frontend (React + Vite, port 3000)
```bash
npm install
npm run dev -- --host 0.0.0.0 --port 3000
```
Or double-click `start_frontend.bat`. Open `http://localhost:3000`.

### LAN demo
Start both commands above, then open `http://<this-PC-LAN-IP>:3000` from another device on the
same network (allow ports 3000 and 8000 through the Windows firewall). By default the frontend calls
`http://<host the page was opened from>:8000`, so no IP needs to be hardcoded. To point at a backend
on another host, set `VITE_API_BASE_URL` in `.env.local` (see `.env.example`).

### Tests
```bash
python -m pytest -q                       # all suites in tests/
python -m pytest tests/test_argo_api.py -v
python -m compileall backend src -q
npm run lint                              # tsc --noEmit
npm run build
```

---

## API endpoints

| Method | Path | Purpose |
|:--|:--|:--|
| GET | `/` | Service name and docs link |
| GET | `/health` | Service status, model loaded, device |
| GET | `/model-info` | Architecture, parameters, inputs, depths, domain, checkpoint |
| GET | `/available-dates` | The 274 processed dates (`?format=list` for a plain array) |
| POST | `/predict` | `{date, latitude, longitude}` → 15-depth profile, surface inputs, MLD, thermocline depth, OHC300 |
| GET | `/argo/profiles` | The 36-profile ARGO evaluation catalog (metadata) |
| GET | `/argo/summary` | Aggregate ARGO evaluation computed at request time (`?refresh=true` to recompute) |
| GET | `/argo/compare/{profile_id}` | One ARGO profile vs the OceanEmbed reconstruction at its date and grid cell |

Errors: 400 for out-of-domain coordinates or land/missing-SST cells, 404 for unavailable dates or
unknown ARGO profiles, 422 for malformed requests, 503 when the ARGO archive is unavailable.

---

## Repository layout

```
backend/        FastAPI app (main.py), inference service, ARGO evaluation service, schemas, config
src/            Python ML core (models/, preprocessing/, validation/argo_catalog.py, evaluation/)
                and the React frontend (App.tsx, components/, services/api.ts, types/)
checkpoints/    phase5/oceanembed_v3_decoder.pt (served model)
configs/        train-only scaler + provenance metadata
data/processed/ monthly Jan–Sep 2020 tensor chunks (inputs + GLORYS targets)
data/argo/      argo2020/ Sep 2020 GDAC NetCDF snapshots (evaluation only)
reports/        benchmark JSON, ARGO matching ledger, audits (reports/final/)
scripts/        offline data-pipeline and evaluation scripts
tests/          pytest suites (backend, ARGO API, leakage, legacy e2e)
```

## Scope and limitations
* Region and period are limited to the North Indian Ocean and January–September 2020.
* No live satellite ingestion, no live ARGO feed, and no forecasting: the model reconstructs the
  subsurface state for dates whose surface fields are already in the processed archive.
* The ARGO evaluation covers three September 2020 snapshot days (36 profiles). Generalisation to
  other seasons and years has not been evaluated observationally.
* The thermocline band shading (75–150 m) is a fixed display band; the per-profile thermocline
  depth is computed from the reconstructed profile.

Detailed audit: `reports/final/FINAL_MVP_AUDIT.md`.

---

## Team & Presentation
- **Project Name:** OceanEmbed
- **Hackathon:** Smart India Hackathon (SIH 2026)
- **Problem Statement:** SIH26066 — Reconstruction of Subsurface Ocean Temperature
- **Lead / Developer:** Ankit Soni (`ankitsoni3874@gmail.com`)
