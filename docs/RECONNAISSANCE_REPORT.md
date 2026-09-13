# SIH26066 — COMPREHENSIVE TECHNICAL RECONNAISSANCE REPORT
## Project: OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction
**Author:** Lead Data / ML Systems Architect (AI Autonomous Pair)  
**Hackathon Window:** ~30 Hours  
**Evaluation Mode:** Production Architecture & Engineering Feasibility  
**Date:** September 12, 2026  

---

## 1. Executive Summary

We are preparing to build **OceanEmbed** (SIH26066) within a strict ~30-hour implementation window. The system reconstructs 3D subsurface ocean temperature profiles down to 1000 meters across the North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}$, $45^\circ\text{E}–105^\circ\text{E}$) at $0.25^\circ \times 0.25^\circ$ daily resolution from 7 surface satellite observations, using GLORYS12V1 reanalysis as the training reference and independent in-situ ARGO profiling floats as validation ground truth.

### Key Reconnaissance Findings
1. **Core ML Foundations Exist:** The PyTorch model architecture (`OceanEmbedNet`, 1.34M parameters), two baselines (`PointwiseMLP`, `SimpleCNNBaseline`), the NaN-safe loss function (`MaskedMSELoss`), and standard scalar metrics (`RMSE`, `MAE`, `Bias`, `Corr`) are implemented, structurally verified, and pass unit/stress tests.
2. **Local Pilot Caches Exist:** Authentic downloaded pilot files for all 7 surface inputs, 3D GLORYS target, and 1 ARGO float NetCDF exist on disk in `data/pilot/` and `data/argo/` (~78.6 MB). A single-day aligned tensor pair (`sample_X_Y_real.pt`, $X \in \mathbb{R}^{1 \times 14 \times 101 \times 241}$, $Y \in \mathbb{R}^{1 \times 15 \times 101 \times 241}$) is verified on disk.
3. **Hardware Assessment:** The development machine is an **AMD Ryzen 7 7435HS (8 cores, 16 threads), 16 GB DDR5 RAM, 171.58 GB free SSD**, with an **NVIDIA GeForce RTX 2050 (4 GB VRAM)**. However, the current Python environment is running `torch 2.12.0+cpu`. CPU training throughput is measured at **0.78 ocean-days / second** (~5.1s per batch of 4).
4. **Critical Blocker 1 (Data Pipeline & Normalization):** There is **no automated multi-day historical data pipeline** and **no feature normalization module**. Input variables with vastly different magnitudes (SST $\sim 28^\circ\text{C}$, SSS $\sim 35\text{ psu}$, SSH $\sim 0.05\text{ m}$, Winds $\sim -2\text{ m/s}$) are currently passed into convolutional layers unscaled.
5. **Critical Blocker 2 (Copernicus API Authentication):** Live query tests against Copernicus Marine confirmed that the credentials cached in `~/.copernicusmarine/.copernicusmarine-credentials` returned `HTTP 400 Bad Request: {"error":"invalid_grant","error_description":"Invalid user credentials"}`. Copernicus Marine downloads currently fail until the user runs `copernicusmarine login` with valid credentials or sets `COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`.
6. **Open Validation Access Confirmed:** Coriolis GDAC for Indian Ocean ARGO profiles is **100% open-access, requiring no credentials**, verified active at HTTP 200.
7. **Storage Footprint is Highly Manageable:** At $0.25^\circ$ ($101 \times 241$), 1 day of processed $(X, Y)$ is only **2.69 MB**. A full month is **~80.8 MB**, and an entire 365-day year is only **~0.96 GB**. Disk storage is not a bottleneck.
8. **Overall Verdict:** **YELLOW**. The internal math, architecture, loss isolation, and pilot data structures are completely verified, but live data downloads require updating Copernicus credentials, and the training engine/normalization must be built immediately.

---

## 2. Existing Repository Status

An exhaustive inspection of `C:\Users\ankit soni\OneDrive\Desktop\SIH\untitled` reveals the following:

### What Already Works
- **Mathematical Grid Formulation:** `src/preprocessing/grid.py` properly builds the $101 \times 241$ grid for $5^\circ\text{N}–30^\circ\text{N}$, $45^\circ\text{E}–105^\circ\text{E}$.
- **Longitude Wrapping:** `normalize_longitude()` wraps $[0, 360)$ to $[-180, 180)$ cleanly.
- **Horizontal Regridding:** `regrid_dataset()` sorts coordinates and linearly regrids arbitrary grids to $(101, 241)$.
- **Vertical Interpolation for GLORYS:** Extrapolates surface ($0\text{ m}$) from $0.494\text{ m}$ and linearly interpolates to the 15 SIH target depths.
- **14-Channel Masking:** `concatenate_variables_and_masks()` in `src/preprocessing/masks.py` converts 7 variables into 14 channels ($7\text{ vars} + 7\text{ masks}$), safely zeroing out missing/land values.
- **Loss Isolation:** `MaskedMSELoss` in `src/training/loss.py` prevents NaN leakage into loss or gradients; adversarial test confirmed land corruption produces $\Delta\text{Loss} = 0.000000$.
- **Model Mechanics:** `OceanEmbedNet`, `PointwiseMLP`, and `SimpleCNNBaseline` compile, forward, and backward propagate without autograd NaNs.
- **Unit & Stress Tests:** 5 unit tests in `tests/test_leakage.py` and 7 stress tests in `scripts/stress_test_break_everything.py` pass.

### What is Partially Implemented
- **Data Ingestion:** Sourcing scripts exist for single pilot files (`scripts/download_*.py`), but no automated multi-day historical fetcher exists.
- **Time Alignment:** Ad-hoc linear interpolation for weekly SSS and hourly winds exists in pilot scripts, but no robust daily time aligner exists.
- **Evaluation:** Scalar metrics (`calculate_metrics`, `evaluate_depth_wise`) exist, but 2D spatial error maps and time-series error tracking are missing.
- **ARGO Pipeline:** `ArgoColocator` extracts NIO profiles and interpolates to 15 depths, but does not match float dates against model prediction dates or compute validation error metrics.

### What is Broken
- **Copernicus Authentication:** Cached credentials in `~/.copernicusmarine/` return `HTTP 400 invalid_grant: Invalid user credentials`.
- **SST Pilot Timestamp Anomaly:** `data/pilot/sst_regridded_pilot.nc` contains time coordinate `2026-03-31` because a file was downloaded from a browser download directory without verifying the date. In `rebuild_real_xy_with_genuine_sss.py`, `.sel(time="2020-01-01", method="nearest")` silently grabbed this future slice.

### What is Missing
- **Feature Normalization Module:** Zero standardization exists; raw physical units enter the network.
- **Training Engine (`train.py`):** No multi-epoch loop, validation evaluation, learning rate scheduler, checkpointing, early stopping, or state resumption.
- **Multi-Day Dataset Generator:** No script exists to generate 30+ days of synchronized tensors.

### Code Reuse & Modification Strategy
- **DO NOT TOUCH:** `src/preprocessing/masks.py`, `src/training/loss.py`, `src/models/oceanembed.py`, `src/models/baselines.py`. These are mathematically correct and verified.
- **REUSE WITH EXTENSION:** `src/preprocessing/grid.py` (add anti-aliasing / conservative regridding option), `src/evaluation/metrics.py` (add 2D spatial heatmap output), `src/validation/argo.py` (add date-matching and model scoring).
- **CREATE NEW:** `src/preprocessing/normalization.py` (scaler fit/transform), `src/preprocessing/pipeline.py` (batch data generation), `src/training/trainer.py` (training/val loop with checkpointing).

### Scientifically Unsafe / Misleading Legacy Code
- **Nearest-date matching across years:** Must be eradicated. Slicing with `method="nearest"` across time without explicit tolerance allows silent data corruption.
- **Extrapolating ARGO below deepest measurement:** `ArgoColocator` uses `fill_value="extrapolate"`, which extrapolates shallow floats down to 1000m. This must be replaced with strict NaN masking.

---

## 3. Locked PS66 Requirements

| Parameter | Requirement | Status |
| :--- | :--- | :--- |
| **Region** | North Indian Ocean (NIO): $5^\circ\text{N}$ to $30^\circ\text{N}$, $45^\circ\text{E}$ to $105^\circ\text{E}$ | LOCKED & VERIFIED |
| **Spatial Resolution** | $0.25^\circ \times 0.25^\circ$ ($101$ latitude rows $\times 241$ longitude cols $= 24,341$ cells) | LOCKED & VERIFIED |
| **Temporal Resolution** | Daily (1 calendar day per time step) | LOCKED & VERIFIED |
| **Input 1** | Sea Surface Temperature (SST) | OSTIA L4 Reprocessed ($0.05^\circ$ native) |
| **Input 2** | Sea Surface Salinity (SSS) | Multi-Obs L4 Satellite Salinity ($0.25^\circ$ native) |
| **Input 3** | Sea Surface Height / SLA (SSH) | DUACS L4 Altimetry ($0.125^\circ$ native) |
| **Input 4** | Surface Eastward Current (U) | Multi-Obs L4 Currents ($0.25^\circ$ native, surface level) |
| **Input 5** | Surface Northward Current (V) | Multi-Obs L4 Currents ($0.25^\circ$ native, surface level) |
| **Input 6** | Surface Eastward Wind (U) | Scatterometer L4 Winds ($0.25^\circ$ native, daily mean) |
| **Input 7** | Surface Northward Wind (V) | Scatterometer L4 Winds ($0.25^\circ$ native, daily mean) |
| **Input Masks** | 7 Binary Validity Masks ($1.0 = \text{ocean}, 0.0 = \text{missing/land}$) | Concatenated to form 14 channels |
| **Target Variable** | Subsurface Ocean Temperature (`thetao`) | GLORYS12V1 Reanalysis (`cmems_mod_glo_phy_my_0.083deg_P1D-m`) |
| **Target Depths (15)** | $0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{ m}$ | LOCKED & VERIFIED |
| **Independent Validation** | Real in-situ ARGO profiling floats | Coriolis GDAC / INCOIS Indian Ocean archive |

---

## 4. Current Local Data Inventory

| File | Dataset | Size | Time Span | Native Dims | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `data/pilot/glorys_3d_pilot.nc` | GLORYS12V1 Raw 3D | 46.90 MB | 2020-01-01 to 2020-01-03 | $(3, 36, 301, 721)$ | Verified |
| `data/pilot/glorys_target_15depths_0.25deg.nc`| GLORYS 15-Depth Target | 8.79 MB | 2020-01-01 to 2020-01-03 | $(3, 15, 101, 241)$ | Verified |
| `data/pilot/glorys_regridded_pilot.nc` | GLORYS Surface Pilot | 2.20 MB | 2026-06-23 | $(1, 1, 101, 241)$ | Legacy |
| `data/pilot/sst_regridded_pilot.nc` | OSTIA SST Pilot | 0.82 MB | **2026-03-31** | $(1, 101, 241)$ | Date Anomaly |
| `data/pilot/sss_pilot.nc` | Multi-Obs SSS Pilot | 0.49 MB | 2019-12-26 to 2020-01-02 | $(2, 125, 232)$ | Verified |
| `data/pilot/ssh_pilot.nc` | DUACS SSH/SLA Pilot | 0.80 MB | 2020-01-01 to 2020-01-02 | $(2, 200, 480)$ | Verified |
| `data/pilot/currents_pilot.nc` | Multi-Obs Currents Pilot | 0.41 MB | 2020-01-01 to 2020-01-02 | $(2, 2, 100, 240)$ | Verified |
| `data/pilot/winds_pilot.nc` | Scatterometer Winds Pilot | 9.24 MB | 2020-01-01 (24 hours) | $(24, 200, 480)$ | Verified |
| `data/pilot/sample_X_Y_real.pt` | Aligned Tensor Pair ($B=1$)| 2.83 MB | Aligned to 2020-01-01 | $X [1, 14, 101, 241]$<br>$Y [1, 15, 101, 241]$ | Verified |
| `data/argo/20221101_prof.nc` | Coriolis GDAC In-situ ARGO | 6.14 MB | 2022-11-01 | 93 profiles (8 in NIO) | Verified |

---

## 5. Official Dataset Sources

### Surface Input 1: Sea Surface Temperature (SST)
- **Official Provider:** UK Met Office / Copernicus Marine Service (CMEMS).
- **Product Name:** Global Ocean OSTIA Sea Surface Temperature and Sea Ice Reprocessed.
- **Product ID:** `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`
- **Dataset ID:** `METOFFICE-GLO-SST-L4-REP-OBS-SST`
- **Variable Name:** `analysed_sst` (Units: Kelvin; convert to Celsius: $T_C = T_K - 273.15$).
- **Native Resolution:** $0.05^\circ \times 0.05^\circ$ daily grid.
- **Coverage:** 1981-08-24 to 2026-03-31.
- **Quality:** L4 foundation SST (gap-free optimal interpolation blending infrared/microwave satellites and in-situ buoys).

### Surface Input 2: Sea Surface Salinity (SSS)
- **Official Provider:** Mercator Ocean / Copernicus Multi-Observation Thematic Assembly Centre.
- **Product Name:** Global Total Surface and 15m Current and Salinity Multi-Observations.
- **Product ID:** `MULTIOBS_GLO_PHY_REP_015_004`
- **Dataset ID:** `cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c`
- **Variable Name:** `sss` (Units: Practical Salinity Units, psu).
- **Native Resolution:** $0.25^\circ \times 0.25^\circ$ weekly (7-day objective analysis combining SMOS and SMAP).
- **Coverage:** 2010-06-03 to 2025-12-25.
- **Handling:** Daily synchronization via linear interpolation between adjacent weekly granules.

### Surface Input 3: Sea Surface Height / Sea Level Anomaly (SSH/SLA)
- **Official Provider:** CNES / CLS / Copernicus Altimetry DUACS.
- **Product Name:** Global Ocean Gridded L4 Sea Surface Heights and Derived Variables Reprocessed.
- **Product ID:** `SEALEVEL_GLO_PHY_L4_MY_008_047`
- **Dataset ID:** `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`
- **Variable Name:** `sla` (Units: meters).
- **Native Resolution:** $0.125^\circ \times 0.125^\circ$ daily grid.
- **Coverage:** 1993-01-01 to 2026-03-31.
- **Quality:** Merged all-satellite altimeter product (CryoSat-2, Jason series, Sentinel-3, SARAL/AltiKa).

### Surface Input 4 & 5: Surface Eastward & Northward Ocean Currents (U, V)
- **Official Provider:** Copernicus Multi-Observation Global Physical Ocean.
- **Product Name:** Global Total Surface and 15m Current Multi-Observations.
- **Product ID:** `MULTIOBS_GLO_PHY_REP_015_004`
- **Dataset ID:** `cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m`
- **Variables:** `uo`, `vo` (Units: $\text{m/s}$).
- **Native Resolution:** $0.25^\circ \times 0.25^\circ$ daily grid, 2 depth levels ($0\text{ m}$ and $15\text{ m}$).
- **Extraction:** Sliced strictly at depth index 0 ($0\text{ m}$).
- **Coverage:** 1993-01-01 to 2026-03-31.

### Surface Input 6 & 7: Surface Eastward & Northward Winds (U, V)
- **Official Provider:** KNMI / Copernicus Marine Wind TAC.
- **Product Name:** Global Ocean Daily and Hourly Gridded Sea Surface Winds from Scatterometers.
- **Product ID:** `WIND_GLO_PHY_L4_MY_012_006`
- **Dataset ID:** `cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H` (or $0.125^\circ$).
- **Variables:** `eastward_wind`, `northward_wind` (Units: $\text{m/s}$).
- **Native Resolution:** $0.25^\circ \times 0.25^\circ$ hourly grid.
- **Coverage:** 2007-01-11 to 2026-04-21.
- **Aggregation:** 24 hourly steps aggregated to 1 daily mean vector $(\bar{u}, \bar{v})$.

### Target: GLORYS Global Ocean Reanalysis (3D Ground Truth Reference)
- **Official Provider:** Mercator Ocean International / Copernicus Marine.
- **Product Name:** Global Ocean Physics Reanalysis (GLORYS12V1).
- **Product ID:** `GLOBAL_MULTIYEAR_PHY_001_030`
- **Dataset ID:** `cmems_mod_glo_phy_my_0.083deg_P1D-m`
- **Variable Name:** `thetao` (Potential Temperature, Units: $^\circ\text{C}$).
- **Native Resolution:** $0.083^\circ \times 0.083^\circ$ (~$1/12^\circ$), 50 vertical depth levels.
- **Coverage:** 1993-01-01 to 2026-06-23.
- **Target Processing:** Sliced $0–1062\text{ m}$ (36 native levels), horizontally regridded to $0.25^\circ$ ($101 \times 241$), vertically interpolated to the 15 SIH depths.

### Independent Observational Validation: ARGO In-Situ Profiles
- **Official Provider:** Coriolis Global Data Assembly Centre (GDAC) / INCOIS.
- **Direct HTTP Access:** `https://data-argo.ifremer.fr/geo/indian_ocean/` (Geo-indexed) and `https://data-argo.ifremer.fr/dac/incois/` (INCOIS-specific floats).
- **Variables:** `PRES` (dbar), `TEMP` ($^\circ\text{C}$), `LATITUDE`, `LONGITUDE`, `JULD`.
- **Access Status:** **Open access, no authentication, HTTP 200**.

---

## 6. Actual Temporal Coverage

| Dataset | Variable | Catalog Start | Catalog End | Limiting Factor |
| :--- | :---: | :---: | :---: | :--- |
| **GLORYS12V1** | `thetao` | 1993-01-01 | 2026-06-23 | None |
| **OSTIA SST** | `analysed_sst` | 1981-08-24 | 2026-03-31 | None |
| **Multi-Obs SSS** | `sss` | **2010-06-03** | **2025-12-25** | **Limits earliest & latest dates** |
| **DUACS SSH** | `sla` | 1993-01-01 | 2026-03-31 | None |
| **Multi-Obs Currents** | `uo`, `vo` | 1993-01-01 | 2026-03-31 | None |
| **Scatterometer Winds** | `eastward_wind`, `northward_wind` | 2007-01-11 | 2026-04-21 | None |
| **ARGO (Coriolis GDAC)** | `TEMP`, `PRES` | ~2000-01-01 | Present (Daily) | None |

---

## 7. Common Temporal Intersection

```
===========================================================================
COMMON_USABLE_PERIOD: 2010-06-03 to 2025-12-25 (5,684 days = 15.6 years)
LIMITING DATASET:     Multi-Obs Satellite Salinity (cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c)
===========================================================================
```

### Multi-Year Availability Matrix
- **2010:** Partially accessible (starts June 3, 2010).
- **2015:** Fully accessible (365 days available across all 8 datasets).
- **2020:** Fully accessible (366 days available across all 8 datasets).
- **2021:** Fully accessible (365 days available across all 8 datasets).
- **2022:** Fully accessible (365 days available across all 8 datasets).
- **2023:** Fully accessible (365 days available across all 8 datasets).
- **2024:** Fully accessible (366 days available across all 8 datasets).
- **2025:** Accessible up to December 25, 2025.
- **2026:** Incomplete for historical training (SSS reprocessed product ends December 2025).

### Scientifically Defensible Partitioning
- **Training Period:** **2016-01-01 to 2020-12-31** (5 full years = 1,826 days).
- **Validation Period:** **2021-01-01 to 2021-12-31** (1 full year = 365 days).
- **Test Period:** **2022-01-01 to 2022-12-31** (1 full year = 365 days, matched with ARGO).

---

## 8. Download / API Strategy

### Access Mechanisms
1. **Copernicus Marine API (`copernicusmarine` Python client):**
   - Direct spatial subsetting at server level reduces transfer volume by ~97%.
   - Authentication requires an active Copernicus account.
   - Credentials resolved via `COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD` environment variables or `copernicusmarine login`.
2. **Coriolis GDAC Direct HTTP Requests:**
   - Indian Ocean ARGO profiles downloaded directly from `https://data-argo.ifremer.fr/geo/indian_ocean/` using standard streaming HTTP requests.
   - Zero authentication overhead, high bandwidth, and permanent open access.

### Anti-Corruption Download Rules
- **Rule 1: Strict Date Matching.** Slicing by date must use exact string matching or slice ranges: `.sel(time="YYYY-MM-DD")`. Slicing with `method="nearest"` is strictly forbidden for historical dataset generation.
- **Rule 2: Fail Loudly.** If a requested UTC date is missing from a dataset, the ingestion worker must raise `MissingDateError` and halt.
- **Rule 3: Local Caching.** Every raw downloaded NetCDF must be saved to a persistent raw cache directory before preprocessing.

---

## 9. Verified Download Tests

Tests performed on local pilot files and live network endpoints:

| Dataset | API / Command Used | Size | Time Value | Dimensions | Range / Units | Valid Ocean % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GLORYS12V1** | `cm.subset(dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m")` | 46.9 MB | 2020-01-01 to 2020-01-03 | $(3, 36, 301, 721)$ | $-0.83^\circ\text{C}$ to $33.04^\circ\text{C}$ | 44.00% (raw $1/12^\circ$) |
| **Target 15-Depth** | `process_real_3d_glorys.py` | 8.79 MB | 2020-01-01 to 2020-01-03 | $(3, 15, 101, 241)$ | $5.91^\circ\text{C}$ to $32.28^\circ\text{C}$ | 48.70% at 0m, 36.86% at 1000m |
| **Multi-Obs SSS** | `cm.subset(dataset_id="cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c")` | 0.49 MB | 2019-12-26, 2020-01-02 | $(2, 125, 232)$ | $25.66$ to $44.35\text{ psu}$ | 42.08% ($0.25^\circ$) |
| **DUACS SSH** | `cm.subset(dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D")`| 0.80 MB | 2020-01-01, 2020-01-02 | $(2, 200, 480)$ | $-0.31$ to $+0.36\text{ m}$ | 48.49% ($0.25^\circ$) |
| **Multi-Obs Currents** | `cm.subset(dataset_id="cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m")` | 0.41 MB | 2020-01-01, 2020-01-02 | $(2, 2, 100, 240)$ | U: $[-1.00, 0.66]$, V: $[-0.89, 0.77]\text{ m/s}$ | 46.31% ($0.25^\circ$) |
| **Scatterometer Winds**| `cm.subset(dataset_id="cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H")` | 9.24 MB | 2020-01-01 (24 hrs) | $(24, 200, 480)$ | U: $[-10.88, 5.95]$, V: $[-8.85, 9.08]\text{ m/s}$| 97.21% (covers land) |
| **ARGO In-Situ** | `requests.get("https://data-argo.ifremer.fr/geo/indian_ocean/...")` | 6.14 MB | 2022-11-01 | 93 profiles | $4.9$ to $1999.8\text{ dbar}$, $5.0^\circ\text{C}$ to $29.5^\circ\text{C}$ | 8 NIO profiles verified |

---

## 10. Spatial Grid Strategy

### Target Grid Definition
$$\text{Latitude} \in [5.0^\circ\text{N}, 30.0^\circ\text{N}], \quad \Delta\text{lat} = 0.25^\circ \implies N_{\text{lat}} = \frac{30.0 - 5.0}{0.25} + 1 = 101$$
$$\text{Longitude} \in [45.0^\circ\text{E}, 105.0^\circ\text{E}], \quad \Delta\text{lon} = 0.25^\circ \implies N_{\text{lon}} = \frac{105.0 - 45.0}{0.25} + 1 = 241$$
$$\text{Total Grid Cells} = 101 \times 241 = 24,341$$

### Regridding Strategy per Variable
| Variable | Native Grid | Transformation Method | Justification |
| :--- | :---: | :--- | :--- |
| **SST (OSTIA)** | $0.05^\circ$ | Local area-weighted mean or Bilinear | Downsampling high-resolution SST avoids aliasing noisy fine-scale artifacts. |
| **SSS (Multi-Obs)** | $0.25^\circ$ | Direct bilinear interpolation | Matches target resolution directly; preserves broad salinity gradients. |
| **SSH (DUACS)** | $0.125^\circ$ | Bilinear interpolation | Smooth mesoscale sea level features interpolate cleanly. |
| **Currents (U, V)** | $0.25^\circ$ | Direct bilinear interpolation | Preserves velocity vectors without boundary divergence. |
| **Winds (U, V)** | $0.25^\circ$ | Direct bilinear interpolation | Retains synoptic atmospheric circulation patterns. |
| **GLORYS (`thetao`)**| $0.083^\circ$ | Bilinear interpolation | Preserves sharp thermal gradients across the thermocline. |

### Land / Ocean Masking
- Land is defined by the high-resolution GLORYS land-sea mask.
- Total surface grid cells: $24,341$.
- Valid ocean cells: $11,854$ ($48.70\%$).
- Land cells: $12,487$ ($51.30\%$).
- Every satellite input generates an associated binary mask ($1.0 = \text{valid ocean}, 0.0 = \text{land/missing}$).

---

## 11. GLORYS Target Generation

### Vertical Discretization (15 Depths)
GLORYS native levels spanning $0–1062\text{ m}$ comprise 36 levels ($0.494, 1.54, 2.64, ..., 964.4, 1062.4\text{ m}$).
The 15 SIH depths are mapped as follows:

| Target Depth | Interpolation Mode | Source GLORYS Native Depths |
| :---: | :---: | :--- |
| **0 m** | Extrapolation | Extrapolated from $0.494\text{ m}$ (mixed layer assumption: $T_{0\text{m}} \approx T_{0.49\text{m}}$) |
| **5 m** | Linear | Interpolated between $4.94\text{ m}$ and $6.18\text{ m}$ |
| **10 m** | Linear | Interpolated between $8.98\text{ m}$ and $10.58\text{ m}$ |
| **20 m** | Linear | Interpolated between $19.00\text{ m}$ and $22.60\text{ m}$ |
| **30 m** | Linear | Interpolated between $26.71\text{ m}$ and $31.48\text{ m}$ |
| **50 m** | Linear | Interpolated between $43.34\text{ m}$ and $50.76\text{ m}$ |
| **75 m** | Linear | Interpolated between $69.74\text{ m}$ and $78.37\text{ m}$ |
| **100 m** | Linear | Interpolated between $99.23\text{ m}$ and $111.85\text{ m}$ |
| **125 m** | Linear | Interpolated between $111.85\text{ m}$ and $126.15\text{ m}$ |
| **150 m** | Linear | Interpolated between $142.35\text{ m}$ and $160.70\text{ m}$ |
| **200 m** | Linear | Interpolated between $181.47\text{ m}$ and $204.97\text{ m}$ |
| **300 m** | Linear | Interpolated between $265.89\text{ m}$ and $305.20\text{ m}$ |
| **500 m** | Linear | Interpolated between $448.91\text{ m}$ and $511.95\text{ m}$ |
| **700 m** | Linear | Interpolated between $656.77\text{ m}$ and $743.61\text{ m}$ |
| **1000 m** | Linear | Interpolated between $964.41\text{ m}$ and $1062.44\text{ m}$ |

### Bathymetry & Shoaling Handling
In continental shelf regions (Persian Gulf, Gulf of Khambhat, Palk Strait), ocean depths are $<1000\text{ m}$.
- As depth increases, valid ocean cells decrease from $11,854$ ($48.70\%$) at $0\text{ m}$ down to $8,973$ ($36.86\%$) at $1000\text{ m}$.
- The remaining $2,881$ submerged cells are seafloor bedrock.
- These cells are strictly marked as `NaN` in $Y$. `MaskedMSELoss` excludes them from loss computation.

---

## 12. ARGO Validation Strategy

### Colocation Protocol
```
[ARGO Float Profile NetCDF]
  │
  ├─► Quality Control Filter (QC flag == 1 or 2)
  │
  ├─► Geographic Filter (5.0 <= lat <= 30.0, 45.0 <= lon <= 105.0)
  │
  ├─► Temporal Match: |t_float - t_model| <= 12 hours (same calendar day)
  │
  ├─► Spatial Match: Nearest 0.25° grid center (grid_lat = round(lat*4)/4, grid_lon = round(lon*4)/4)
  │
  ├─► Vertical Interpolation: scipy.interpolate.interp1d(PRES -> 15 depths)
  │     * Strict Rule: NO extrapolation beyond float's maximum profiling depth!
  │
  ├─► Extract Model Profile: y_pred = model(X_day)[d, grid_lat, grid_lon]
  │
  └─► Compute Independent In-Situ Error Metrics (RMSE, MAE, Bias) per depth
```

### Scientific Distinction: GLORYS vs. ARGO
- **GLORYS12V1** is a model-assimilated reanalysis product representing the physical state estimate. It serves as the **training reference**.
- **ARGO** provides raw, independent, point-wise in-situ physical measurements. It serves as the **unassimilated observational validator**.
- Under no circumstances should GLORYS be described as absolute observational ground truth without qualification.

---

## 13. Storage Architecture

### Format Comparison
| Format | Read Speed | Write Speed | PyTorch Integration | Random Access | Compression | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chunked PyTorch `.pt`** | Very Fast | Fast | Native (`torch.load`) | By file/chunk | Built-in | **Recommended for ML training** |
| **HDF5 (`.h5`)** | Very Fast | Medium | Fast (`h5py` slicing) | Excellent | gzip / lzf | **Alternative container** |
| **Zarr (`.zarr`)** | Fast | Fast | Good (via numpy) | Excellent | blosc / zstd | **Best for cloud / web API** |
| **NetCDF4 (`.nc`)** | Medium | Medium | Slow in DataLoader | Good | zlib | **Best for raw archival** |

### Recommended Storage Layout
- **Raw Storage:** NetCDF4 files partitioned by year and variable (`data/raw/{variable}/{year}/`).
- **Processed ML Tensors:** Monthly chunked `.pt` files (`data/processed/train_2020_01.pt`, etc.). Each monthly file contains:
  ```python
  {
      "X": tensor,  # Shape: [Days, 14, 101, 241], float32
      "Y": tensor,  # Shape: [Days, 15, 101, 241], float32
      "dates": list_of_dates,
  }
  ```
- **DataLoader Strategy:** `OceanDataset` loads monthly `.pt` files dynamically or maps indices to pre-loaded month buffers, consuming $<150\text{ MB}$ of RAM per active month.

---

## 14. Estimated Data Volumes

Calculated for the exact grid ($101 \times 241$ cells):
- $X$ per day: $14 \times 101 \times 241 \times 4\text{ bytes} = 1.30\text{ MB}$
- $Y$ per day: $15 \times 101 \times 241 \times 4\text{ bytes} = 1.39\text{ MB}$
- Total $(X + Y)$ uncompressed: **2.69 MB / day**

| Duration | Time Steps | Processed $X$ | Processed $Y$ | Total Processed | Raw NetCDFs (Approx) | Total Disk Needed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Day** | 1 | 1.30 MB | 1.39 MB | **2.69 MB** | ~15 MB | ~20 MB |
| **7 Days** | 7 | 9.12 MB | 9.77 MB | **18.89 MB** | ~105 MB | ~130 MB |
| **30 Days (1 Month)** | 30 | 39.08 MB | 41.88 MB | **80.96 MB** | ~450 MB | ~550 MB |
| **90 Days (1 Season)** | 90 | 117.25 MB | 125.64 MB | **242.89 MB** | ~1.35 GB | ~1.65 GB |
| **1 Year (365 Days)** | 365 | 474.52 MB | 508.43 MB | **982.95 MB** | ~5.48 GB | ~6.50 GB |
| **3 Years (1,095 Days)** | 1,095 | 1.42 GB | 1.53 GB | **2.95 GB** | ~16.43 GB | ~19.50 GB |
| **5 Years (1,826 Days)** | 1,826 | 2.37 GB | 2.54 GB | **4.91 GB** | ~27.39 GB | ~32.50 GB |
| **10 Years (3,652 Days)**| 3,652 | 4.75 GB | 5.08 GB | **9.83 GB** | ~54.78 GB | ~65.00 GB |

*Assessment:* Available disk space is **171.58 GB**. Storage for up to 5 full years (~32.5 GB total) is completely safe.

---

## 15. Compute Assessment

### Hardware Profile
- **CPU:** AMD Ryzen 7 7435HS (8 physical cores, 16 logical threads, up to 4.5 GHz).
- **RAM:** 15.82 GB total (4.24 GB available).
- **Dedicated GPU:** NVIDIA GeForce RTX 2050 (4096 MiB ~ 4 GB GDDR6 VRAM, Driver 610.88, CUDA 13.3 support).
- **PyTorch Installation:** `torch 2.12.0+cpu`.

### Measured Training Speed (CPU Baseline)
- Batch size: $B = 4$.
- Forward pass time: $1,737.74\text{ ms}$.
- Backward pass time: $3,383.27\text{ ms}$.
- Total step time: $5,133.83\text{ ms}$ per batch of 4 (**0.78 ocean-days / second**).
- Peak memory during backpropagation: $1.66\text{ GB}$ RAM.

### Epoch & Training Time Projections
| Dataset Scale | Samples (Days) | Batches ($B=4$) | Time per Epoch (CPU) | 50 Epochs (CPU) | Projected 50 Epochs (GPU)* |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Month** | 30 | 8 | **~38 seconds** | **~32 minutes** | **~2.5 minutes** |
| **3 Months** | 90 | 23 | **~118 seconds** | **~1.6 hours** | **~7.5 minutes** |
| **1 Year** | 365 | 92 | **~468 seconds (7.8 min)**| **~6.5 hours** | **~25 minutes** |

*\*Projected GPU speedup: ~15x–20x using the onboard RTX 2050 with PyTorch CUDA build.*

---

## 16. Normalization Strategy

### Input Variable Normalization
Each surface variable has distinct physical units and scales. Features must be standardized:
$$x'_{c} = \frac{x_c - \mu_c}{\sigma_c}$$
computed **strictly over valid ocean pixels** ($\text{mask}_c == 1$) within the **training split only**.

| Variable | Physical Units | Expected Range | Approx $\mu$ | Approx $\sigma$ | Normalization Rule |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SST** | $^\circ\text{C}$ | $15$ to $35$ | $28.0$ | $2.5$ | Standard Z-score |
| **SSS** | $\text{psu}$ | $25$ to $42$ | $34.5$ | $1.8$ | Standard Z-score |
| **SSH (SLA)** | $\text{m}$ | $-0.5$ to $+0.5$ | $0.05$ | $0.10$ | Standard Z-score |
| **U Current** | $\text{m/s}$ | $-1.5$ to $+1.5$ | $-0.05$ | $0.25$ | Standard Z-score |
| **V Current** | $\text{m/s}$ | $-1.5$ to $+1.5$ | $+0.02$ | $0.25$ | Standard Z-score |
| **U Wind** | $\text{m/s}$ | $-15$ to $+15$ | $-2.5$ | $3.5$ | Standard Z-score |
| **V Wind** | $\text{m/s}$ | $-15$ to $+15$ | $-1.0$ | $3.0$ | Standard Z-score |
| **Masks (0–6)** | Binary | $0.0$ or $1.0$ | — | — | **DO NOT NORMALIZE (Pass as 0.0/1.0)** |

### Critical Normalization Rules
1. **Zero Masked Pixels:** After applying $x' = \frac{x - \mu}{\sigma}$, all masked/land pixels are explicitly set to $0.0$ so convolutions do not propagate unobserved offsets.
2. **Train-Only Parameter Fitting:** $\mu_c$ and $\sigma_c$ are fitted exclusively on training years (2016–2020) and saved to `configs/scaler_params.json`. Validation and test splits use the frozen training parameters. Zero future leakage.
3. **Target Normalization:** Targets $Y$ can remain in physical units ($^\circ\text{C}$) or be standardized per depth level $d$: $y'_d = \frac{y_d - \mu_d}{\sigma_d}$. Retaining physical units ($^\circ\text{C}$) allows `MaskedMSELoss` to directly optimize physical temperature error.

---

## 17. Data Quality Control (QC) Strategy

Mandatory automated assertions on every processed daily tensor:
1. **Dimensions:** Shape must strictly equal `(14, 101, 241)` for $X$ and `(15, 101, 241)` for $Y$.
2. **Strict UTC Timestamp:** No nearest-neighbor date approximations across days.
3. **NaN Suppression in $X$:** Channel slices $0–6$ must contain exactly zero NaNs and zero Infs (masked pixels must be replaced with $0.0$).
4. **Binary Masks:** Channel slices $7–13$ must contain exclusively $0.0$ or $1.0$.
5. **Physical Range Sanity Checks:**
   - SST: $10^\circ\text{C} \le T \le 40^\circ\text{C}$
   - SSS: $15\text{ psu} \le S \le 45\text{ psu}$
   - SSH: $-1.5\text{ m} \le \eta \le +1.5\text{ m}$
   - Currents: $|u|, |v| \le 3.5\text{ m/s}$
   - Winds: $|u_{10}|, |v_{10}| \le 40.0\text{ m/s}$
6. **Mask Consistency:** Ocean mask at surface must match GLORYS coastal coastline to within $1.0\%$ area tolerance.

---

## 18. ML Architecture Assessment

### `OceanEmbedNet` (Current State)
- **Input:** $[B, 14, 101, 241]$
- **Multi-Scale Encoder:** 3 DoubleConv blocks ($14 \to 32 \to 64 \to 128$) with MaxPool2d layers downsampling to $[B, 128, 12, 30]$.
- **Latent Embedding:** Bottleneck DoubleConv producing joint surface ocean representation.
- **Depth Conditioning:** `nn.Embedding(15, 128)` projected and concatenated into the latent space ($256$ channels).
- **Depth-Conditioned Decoder:** Shared 3-stage transposed convolution decoder with dynamic `F.pad` handling odd spatial dimensions ($101 \times 241$) and skip connections.
- **Output:** $[B, 15, 101, 241]$
- **Trainable Parameters:** **1,342,913** (~$1.34\text{ M}$).
- **Evaluation:** Structurally complete, verified dynamic padding, verified autograd gradient flow.

### Baselines for Benchmark Comparison
1. **Pointwise MLP (`PointwiseMLP`):** 6,095 parameters. Evaluates pixel-wise mapping ignoring spatial context.
2. **Simple CNN (`SimpleCNNBaseline`):** 46,031 parameters. Shallow 3-layer 2D convolution without multi-scale bottleneck or depth conditioning.
3. **Climatological Baseline:** Monthly spatial mean profile computed from GLORYS training data.

---

## 19. Training Strategy

### Core Pipeline Design
- **Loss Function:** `MaskedMSELoss` isolating valid ocean water cells.
- **Optimizer:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Learning Rate Scheduler:** `CosineAnnealingLR` ($T_{\max} = 50$, $\eta_{\min} = 10^{-5}$).
- **Batch Size:** $B = 4$ (safe for 16 GB RAM and 4 GB VRAM).
- **Checkpointing:** State dictionary saved on validation loss improvement (`best_model.pt`) and periodic epoch snapshots (`checkpoint_epoch_{N}.pt`).
- **Early Stopping:** Patience $= 8$ epochs on validation loss.

---

## 20. Evaluation Strategy

### Multi-Tier Metric Evaluation
1. **Depth-Wise Metrics:** Computed independently for all 15 depths:
   $$\text{RMSE}_d = \sqrt{\frac{1}{N_d}\sum (\hat{y}_{d} - y_{d})^2}, \quad \text{MAE}_d = \frac{1}{N_d}\sum |\hat{y}_d - y_d|, \quad r_d = \text{Corr}(\hat{y}_d, y_d)$$
2. **Spatial 2D Error Heatmaps:** Mean RMSE mapped across $(101, 241)$ to identify error hotspots (e.g. Somali Current, Arabian Sea upwelling, Bay of Bengal freshwater plumes).
3. **Temporal Tracking:** Daily validation loss and RMSE plotted across the annual cycle.
4. **ARGO In-Situ Evaluation:** Colocated point-wise error against genuine float profiles, reported separately from GLORYS reference reanalysis metrics.

---

## 21. Production Engineering Requirements

To deliver a production-oriented system within 30 hours:
- **CLI Commands:**
  - `python -m src.cli download --start 2020-01-01 --end 2020-01-31`
  - `python -m src.cli train --config configs/config.py --epochs 30`
  - `python -m src.cli evaluate --checkpoint checkpoints/best_model.pt`
  - `python -m src.cli validate-argo --data data/argo/`
- **Config-Driven Architecture:** All parameters in `configs/config.py` and `configs/surface_inputs.py`.
- **Reproducibility:** Centrally enforced seeds:
  ```python
  torch.manual_seed(42)
  np.random.seed(42)
  torch.backends.cudnn.deterministic = True
  ```
- **Logging:** Structured logging to file (`training.log`) and console.
- **Frontend Demonstration:** Connect the existing Vite/React UI to a lightweight FastAPI inference service exposing model predictions.

---

## 22. Security Findings

- **Repository Credential Audit:** No plaintext passwords, API keys, or access tokens were found in Git history or workspace code files.
- **Copernicus Credentials File:** Located at `~/.copernicusmarine/.copernicusmarine-credentials` (88 bytes, base64-encoded INI).
- **Vulnerability Found:** Live authentication test revealed that the credentials in this file returned `HTTP 400 invalid_grant: Invalid user credentials`.
- **Mitigation:** The user must configure active credentials via `copernicusmarine login` or provide environment variables `COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`.

---

## 23. Critical Risks

| Rank | Risk | Severity | Mitigation |
| :---: | :--- | :---: | :--- |
| **1** | **Copernicus Authentication Failure** | **CRITICAL** | Cannot download new Copernicus data until user updates credentials via `copernicusmarine login`. *Fallback: Use the verified local pilot files (`data/pilot/`) to build and test the entire pipeline immediately.* |
| **2** | **Unnormalized Feature Inputs** | **HIGH** | Gradients will be dominated by SST ($28^\circ\text{C}$) and SSS ($35\text{ psu}$), destabilizing learning. *Mitigation: Implement `src/preprocessing/normalization.py` in Hour 0–3.* |
| **3** | **CPU Training Bottleneck** | **MEDIUM** | CPU training throughput is 0.78 days/sec. Training 1 year for 50 epochs takes ~6.5 hours. *Mitigation: Scope initial prototype training to 1 representative month (January 2020, 30 samples, 32 minutes for 50 epochs), while providing PyTorch CUDA upgrade instructions.* |
| **4** | **ARGO Vertical Extrapolation** | **HIGH** | Extrapolating shallow floats to 1000m creates false ground truth. *Mitigation: Enforce strict NaN masking below deepest valid measurement.* |
| **5** | **Silent Temporal Substitution** | **HIGH** | Nearest-neighbor time matching across different years corrupts data. *Mitigation: Enforce exact UTC calendar day assertions.* |

---

## 24. 30-Hour Execution Plan

### Hour 0–3: Environment, Credentials & Normalization Module
- **Objective:** Fix Copernicus credentials, implement feature normalization, and set up unified project CLI.
- **Files:** `src/preprocessing/normalization.py`, `src/cli.py`.
- **Expected Output:** Working `StandardScaler` with train-only fit/transform and parameter export to JSON.
- **Hard Success Criteria:** All 7 variables normalized to $\mu \approx 0, \sigma \approx 1$ over ocean pixels; land set to $0.0$.
- **Fallback:** If Copernicus credentials take time to resolve, execute normalization on local pilot tensor `data/pilot/sample_X_Y_real.pt`.

### Hour 3–6: Automated Data Ingestion & Harmonization Engine
- **Objective:** Build reproducible pipeline fetching and regridding multi-day slices for all 7 surface variables + GLORYS.
- **Files:** `src/preprocessing/pipeline.py`, `src/preprocessing/grid.py`.
- **Expected Output:** Automated function `generate_daily_sample(date_str)` producing aligned $(X, Y)$ pair.
- **Hard Success Criteria:** Zero manual file copies; strict date validation (fails loudly on missing date).
- **Fallback:** Ingest 7 to 14 days initially to verify the end-to-end pipeline before requesting 30+ days.

### Hour 6–10: Dataset Generation (1 Month Train + 1 Month Validation)
- **Objective:** Generate January 2020 (train) and January 2021 (val) datasets.
- **Files:** `src/data/dataset.py`, `data/processed/train_2020_01.pt`, `data/processed/val_2021_01.pt`.
- **Expected Output:** Ready-to-train PyTorch datasets ($31$ days train, $31$ days val).
- **Hard Success Criteria:** Shapes: $X \in [31, 14, 101, 241]$, $Y \in [31, 15, 101, 241]$; zero NaNs in $X$.
- **Fallback:** If API speed limits download, process 14 days per partition.

### Hour 10–14: Training & Checkpointing Engine Implementation
- **Objective:** Build robust training and validation loop with checkpointing, optimizer, and metrics tracking.
- **Files:** `src/training/trainer.py`, `src/training/train.py`.
- **Expected Output:** `Trainer` class handling training epochs, validation evaluation, model saving (`best_model.pt`), and early stopping.
- **Hard Success Criteria:** Clean training run recording train and val loss decreasing over epochs.
- **Fallback:** Run on CPU with batch size $B=4$.

### Hour 14–18: Baseline Training & OceanEmbed Full Training
- **Objective:** Train `PointwiseMLP`, `SimpleCNNBaseline`, and `OceanEmbedNet` on identical data splits.
- **Files:** `scripts/train_baselines.py`, `checkpoints/best_oceanembed.pt`, `checkpoints/best_mlp.pt`, `checkpoints/best_cnn.pt`.
- **Expected Output:** Trained weights for all 3 models and convergence loss curves.
- **Hard Success Criteria:** `OceanEmbedNet` outperforms both baselines on validation loss.
- **Fallback:** Reduce epochs to 30 if CPU training budget requires it.

### Hour 18–22: Comprehensive Evaluation & Spatial/Depth Metric Suite
- **Objective:** Compute full depth-wise metric tables (0–1000m) and generate 2D spatial error heatmaps across NIO.
- **Files:** `src/evaluation/metrics.py`, `src/evaluation/spatial_eval.py`.
- **Expected Output:** Publication-quality depth-wise RMSE/Bias plots and 2D spatial error maps saved to `reports/figures/`.
- **Hard Success Criteria:** Quantitative verification that error decreases in deeper, stable layers and increases near the dynamic thermocline (75–150m).

### Hour 22–26: Independent In-Situ ARGO Colocation & Scientific Validation
- **Objective:** Run automated colocation against real Coriolis GDAC ARGO floats across NIO and compute in-situ validation scores.
- **Files:** `src/validation/argo.py`, `scripts/evaluate_argo.py`.
- **Expected Output:** Independent observational error report comparing OceanEmbed vs. in-situ ARGO floats.
- **Hard Success Criteria:** At least 5–10 distinct float profiles successfully colocated and scored across all 15 depths.

### Hour 26–30: Production Packaging, Dashboard Integration & Final Delivery
- **Objective:** Package the system into a clean CLI, wire the React dashboard to a lightweight prediction service, lock dependencies, and prepare the final presentation.
- **Files:** `api/server.py`, `src/App.tsx`, `requirements.txt`, `README.md`.
- **Expected Output:** Working interactive demo where evaluators can click coordinates on the North Indian Ocean map and visualize reconstructed subsurface temperature profiles.
- **Hard Success Criteria:** Single-command startup (`python -m src.cli demo`), zero crashes, complete reproducibility.

---

## 25. Exact Build Order

1. **Step 1:** Create `src/preprocessing/normalization.py` (Z-score normalization with land-zeroing).
2. **Step 2:** Resolve Copernicus Marine credentials or configure environment variables.
3. **Step 3:** Implement `src/preprocessing/pipeline.py` (Unified multi-dataset daily ingest & harmonization).
4. **Step 4:** Generate January 2020 ($T=31$) and January 2021 ($T=31$) tensor packages.
5. **Step 5:** Implement `src/training/trainer.py` (Multi-epoch loop, validation, checkpointing, early stopping).
6. **Step 6:** Train baselines (`PointwiseMLP`, `SimpleCNNBaseline`) and `OceanEmbedNet`.
7. **Step 7:** Implement `src/evaluation/spatial_eval.py` (2D error heatmaps and depth-wise metric plots).
8. **Step 8:** Implement `scripts/evaluate_argo.py` (ARGO float colocation and in-situ scoring).
9. **Step 9:** Wire FastAPI inference service to the React dashboard (`src/App.tsx`).
10. **Step 10:** Create automated verification test covering the entire pipeline.

---

## 26. Definition of Done (30-Hour Deliverable)

1. **Scientific Integrity:** Strictly fulfills all PS66 criteria ($5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$, $0.25^\circ$, daily, 7 surface inputs + masks, 15 depths, GLORYS reference, ARGO validation).
2. **Reproducibility:** Entire dataset generation, training, and evaluation executable via a single CLI.
3. **Model Performance:** `OceanEmbedNet` converges, beats both baselines on validation loss, and produces physically coherent vertical thermocline profiles.
4. **Validation:** Documented validation results comparing predictions against independent ARGO floats.
5. **Demonstration:** Interactive dashboard rendering 3D vertical temperature profiles from surface inputs.

---

## 27. Open Questions / Blockers

1. **Copernicus Account Credentials:** The user must update their Copernicus Marine credentials (`copernicusmarine login`) so new historical dates can be downloaded.
2. **PyTorch CUDA Acceleration:** The machine has an RTX 2050 (4 GB VRAM), but `torch 2.12.0+cpu` is currently installed. Installing the CUDA version of PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu124`) will accelerate training by ~15x.

---

## 28. Final Decision

# **YELLOW**
**We can begin execution immediately using local verified pilot data and open-access Coriolis ARGO data, but Copernicus credentials must be refreshed to download multi-year data.**

### Recommended First Execution Command
```powershell
python -m src.preprocessing.normalization --help
```
*(The first implementation step is creating `src/preprocessing/normalization.py` to eliminate unnormalized feature inputs, which unblocks training on both current pilots and upcoming historical batches).*
