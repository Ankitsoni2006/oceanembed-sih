# OceanEmbed — SIH26066
### Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-4.0-38b2ac.svg)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Executive Summary

**OceanEmbed** is an operational deep learning system developed for **Smart India Hackathon (SIH26066)** that reconstructs 3D subsurface ocean temperature fields across the **North Indian Ocean (5°N–30°N, 45°E–105°E)** down to **1,000 meters depth** exclusively from satellite-derived surface observations.

By pairing surface satellite measurements (SST, SSS, SSH, geostrophic current velocities, and 10m wind fields) with a multiscale spatial-temporal CNN architecture (`OceanEmbedNetV3_Decoder`), OceanEmbed captures wind-driven Ekman pumping, thermocline shoaling, and mesoscale eddy dynamics at **35.9 ms** inference latency per daily basin grid.

The system is rigorously validated against both **CMEMS GLORYS12V1 high-resolution ocean reanalysis** and **in-situ autonomous ARGO profiling floats**, demonstrating physically consistent vertical temperature profiles and thermocline structure.

---

## Key System Highlights & Verified Benchmarks

| Metric / Attribute | Baseline Climatology | Pointwise MLP | Simple Spatial CNN | OceanEmbed (V3 Decoder) | Advantage / Note |
|:---|:---:|:---:|:---:|:---:|:---:|
| **GLORYS Test RMSE (3D Basin)** | 1.6212 °C | 1.6277 °C | 1.1578 °C | **0.8601 °C** | **46.9% error reduction** vs climatology |
| **In-Situ ARGO Float RMSE** | 1.5420 °C | 1.5810 °C | 1.0930 °C | **0.7973 °C** | Validated on 99 independent floats |
| **In-Situ ARGO Correlation ($r$)** | 0.8840 | 0.8790 | 0.9320 | **0.9701** | High vertical coherence |
| **Thermocline RMSE (50–200 m)** | 1.9791 °C | 1.9850 °C | 1.4820 °C | **1.0965 °C** | Resolves sharp vertical gradients |
| **Model Parameters** | — | ~20 K | ~180 K | **1,275,934** | Lightweight (5.1 MB weights) |
| **Inference Latency** | — | — | — | **35.9 ms / day** | **7.2× faster** than 250 ms target |

> **Audit Status:** Rigorously audited via `Phase 7A System Audit` with 100% test pass rate (19/19 pytest suite). Zero data leakage between training (Jan–Aug 2020) and testing (Sep 2020).

---

## System Architecture

```
                                  [SURFACE SATELLITE INPUTS]
            SST + SSS + SSH + Surface Currents (U, V) + 10m Winds (U, V)
                       + Normalized Coordinates (Lat, Lon, Day-of-Year)
                                 [14 Channels × 101 × 241]
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    Spatial Feature Extractor   │
                             │  3-stage multi-receptive conv  │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │  Multiscale Embedding Bottleneck │
                             │   128-dim latent ocean state   │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │   Hierarchical Depth Decoder   │
                             │ Transposed Convolutions + Skips│
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                               [3D SUBSURFACE TEMPERATURE]
                           15 Depth Levels (0 m to 1,000 m)
                                 [15 Depths × 101 × 241]
```

---

## Domain & Vertical Discretization

- **Geographic Coverage:** North Indian Ocean ($5.0^\circ\text{N}$ to $30.0^\circ\text{N}$, $45.0^\circ\text{E}$ to $105.0^\circ\text{E}$)
- **Spatial Resolution:** $0.25^\circ \times 0.25^\circ$ grid ($101 \times 241$ grid cells)
- **Target Depths (15 levels):** `0m, 5m, 10m, 20m, 30m, 50m, 75m, 100m, 125m, 150m, 200m, 300m, 500m, 700m, 1000m`
- **Surface Variables (7 physical channels):**
  1. Sea Surface Temperature (`SST`) [°C]
  2. Sea Surface Salinity (`SSS`) [PSU]
  3. Sea Surface Height Above Geoid (`SSH` / `zos`) [m]
  4. Surface Zonal Current Velocity (`uo`) [m/s]
  5. Surface Meridional Current Velocity (`vo`) [m/s]
  6. 10m Neutral Zonal Wind (`eastward_wind`) [m/s]
  7. 10m Neutral Meridional Wind (`northward_wind`) [m/s]

---

## Repository Structure

```
├── backend/                  # FastAPI inference microservice
│   ├── main.py               # REST API entry point & CORS configuration
│   ├── config.py             # Spatial grid, paths, depth constants
│   ├── inference.py          # PyTorch model loader, LRU cache & engine
│   └── requirements.txt      # Python dependencies
├── checkpoints/              # Model weights & training histories
│   └── phase5/
│       └── oceanembed_v3_decoder.pt   # Frozen production weights (5.1 MB)
├── configs/                  # Normalization scalers & spatial bounds
│   └── scaler_params_experiment_2020.json
├── data/
│   ├── pilot/                # Pilot datasets & verification tensors
│   └── processed/            # Monthly tensor chunks, manifests, climatology
├── docs/                     # Technical specifications, audits & guides
├── reports/                  # Evaluation benchmarks, figures & dossiers
│   └── final/
│       ├── OCEANEMBED_PROJECT_UNDERSTANDING.pdf  # Master technical dossier (6 pages)
│       └── OCEANEMBED_PROJECT_UNDERSTANDING.md
├── src/                      # React 19 + TypeScript + Tailwind frontend
│   ├── components/           # UI components (Map, Profiles, Transects, Metrics)
│   ├── services/api.ts       # Backend REST client
│   └── App.tsx               # Main application layout
├── tests/                    # PyTorch & system integrity test suite
├── start_backend.bat         # One-click Windows launcher for FastAPI
├── start_frontend.bat        # One-click Windows launcher for Vite frontend
└── package.json              # Frontend dependencies & build scripts
```

---

## Quickstart Guide for Team Members

### Prerequisites
- **Python 3.10+** (with `pip`)
- **Node.js 18+** (with `npm`)
- **Git**

### 1. Clone the Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd untitled
```

### 2. Launch the Backend Service
You can use the one-click batch file or launch from terminal:

**Option A (Windows 1-Click):**
Double-click `start_backend.bat`

**Option B (Command Line):**
```bash
# Create and activate virtual environment (optional)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Start FastAPI server (listens on all interfaces at port 8000)
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
- API Swagger Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 3. Launch the Frontend Dashboard
**Option A (Windows 1-Click):**
Double-click `start_frontend.bat`

**Option B (Command Line):**
```bash
# Install frontend packages
npm install

# Start Vite development server
npm run dev
```
- Dashboard Interface: `http://localhost:3000`

---

## Running Automated Tests

Run the complete test suite to verify model dimensions, coordinate indexing, loss calculations, and absence of data leakage:

```bash
pytest tests/ -v
```
All 19 test suites must report `PASSED`.

---

## API Endpoints Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | System health check, device status, and date range |
| `GET` | `/api/dates` | List of all indexed dates with observation chunks |
| `GET` | `/api/depths` | Returns the 15 vertical depth levels in meters |
| `POST` | `/api/predict` | Computes full 3D temperature field or profile at `(lat, lon, date)` |
| `POST` | `/api/spatial-slice` | Returns 2D horizontal temperature grid at a specified depth level |
| `POST` | `/api/transect` | Vertical cross-section along a user-specified lat/lon slice |
| `GET` | `/api/metrics` | Benchmarks, ARGO in-situ validation scores & latency metrics |

---

## Master Technical Documentation

For in-depth mathematical formulations, loss balancing, physical justification, and comprehensive jury defense prep, refer to:
- **Master Technical Dossier (PDF):** [`reports/final/OCEANEMBED_PROJECT_UNDERSTANDING.pdf`](reports/final/OCEANEMBED_PROJECT_UNDERSTANDING.pdf)
- **Markdown Version:** [`reports/final/OCEANEMBED_PROJECT_UNDERSTANDING.md`](reports/final/OCEANEMBED_PROJECT_UNDERSTANDING.md)

---

## Team & Presentation
- **Project Name:** OceanEmbed
- **Hackathon:** Smart India Hackathon (SIH 2026)
- **Problem Statement:** SIH26066 — Reconstruction of Subsurface Ocean Temperature
- **Lead / Developer:** Ankit Soni (`ankitsoni3874@gmail.com`)
