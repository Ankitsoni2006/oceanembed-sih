# OceanEmbed: Technical Understanding Dossier & SIH 2026 Master Reference

**Document Version:** 1.0 (Final Frozen State)  
**Project:** OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations  
**Target Domain:** North Indian Ocean (5°N–30°N, 45°E–105°E)  
**Target Resolution:** 0.25° × 0.25° (101 × 241 grid) | Daily  
**Vertical Strata:** 15 Subsurface Depths (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m)  
**Associated PDF:** `reports/final/OCEANEMBED_PROJECT_UNDERSTANDING.pdf` (6 Pages)

---

## Page 1: Project Overview & Operational Context

### 1. The Operational Challenge
Satellites provide continuous, basin-wide monitoring of surface ocean properties (SST, SSS, SSH, surface winds, currents). However, vital marine physical processes—including the **main thermocline**, upper-ocean heat content (OHC), tropical cyclone heat potential, and underwater acoustic sound ducts—are governed by the **subsurface vertical temperature structure down to 1000 meters**.

In-situ vertical observation systems (such as ARGO profiling floats, CTD casts, and moored buoys) provide high vertical accuracy but are spatially sparse and temporally intermittent. **OceanEmbed** resolves this fundamental observational gap by establishing a deep-learning framework that learns the non-linear physical mapping from surface satellite observation patterns to the complete 15-depth vertical temperature column on a continuous daily 0.25° grid across the North Indian Ocean.

### 2. End-to-End Reconstruction Pipeline
```
1. SURFACE OBSERVATIONS & QUALITY MASKS (14 Channels)
   ├── 7 Physical Surface Observations (SST, SSS, SSH, U/V Currents, U/V Winds)
   └── 7 Binary Validity Quality Masks (1.0 = valid ocean, 0.0 = missing/land)
         │
         ▼
2. MULTI-SCALE SPATIAL ENCODER (Hierarchical U-Net)
   ├── 3-stage DoubleConv blocks (14 -> 32 -> 64 -> 128 channels)
   ├── Multi-scale MaxPool2d downsampling
   └── Lateral feature skip connections for fine spatial details
         │
         ▼
3. LATENT OCEAN EMBEDDING (128-d Bottleneck)
   ├── Compact bottleneck tensor [B, 128, 12, 30]
   ├── Encodes cross-variable coupling and basin-wide spatial equilibrium
   └── Spatial decoder with Transpose-Conv upsampling
         │
         ▼
4. PARALLEL MULTI-DEPTH PROJECTION HEAD
   ├── Dedicated 1x1 convolution mapping 32 features -> 15 depths
   ├── Physical climatological stratification prior
   └── Output: 15-depth subsurface temperature field [B, 15, 101, 241]
```

### 3. Key Quantitative Benchmarks
- **0.8601°C GLORYS Reference RMSE:** Held-out September 2020 test (4,601,790 ocean cells), representing a **-17.44% error reduction vs Simple CNN** (1.0418°C).
- **0.7973°C In-Situ ARGO Observational Check:** Evaluated across 497 matched profile-depth observation pairs from 36 ARGO profiles across 28 floats, achieving a **-16.30% error reduction vs Simple CNN** (0.9526°C).
- **7.2× Training Throughput Speedup:** Parallel multi-depth projection accelerated epoch time from ~172.4 s to ~23.9 s.
- **~35.9 ms Mean API Latency:** Real-time inference on local CPU host (31.85 ms pure neural forward pass).

---

## Page 2: Data Architecture & Problem Formulation

### 1. Mathematical Formulation
Let X_t in R^{14 x H x W} denote the multi-modal gridded surface ocean state at day t, comprising 7 physical variables and 7 binary validity masks over a regular 0.25° grid (H = 101, W = 241). The objective is to estimate the vertical temperature field Y_t in R^{15 x H x W} across 15 standard depths:
D = {0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000} m

The learning objective minimizes masked mean squared error over valid marine grid points:
L(theta) = || M_ocean * (Y - f_theta(X)) ||^2 / || M_ocean ||_1

### 2. Surface Input Data Catalogue & Provenance

| Channel | Physical Variable | Provider & Product Identifier | Native Resolution | Temporal Frequency | Units |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **Ch 0 (Mask 7)** | Sea Surface Temperature (SST) | UK Met Office / OSTIA L4 REP (`010_011`) | 0.05° | Daily | °C |
| **Ch 1 (Mask 8)** | Sea Surface Salinity (SSS) | Copernicus Multi-Obs L4 (`015_004`) | 0.25° | Weekly -> Daily | psu |
| **Ch 2 (Mask 9)** | Sea Surface Height (SSH/SLA) | Copernicus DUACS Altimetry L4 (`008_047`) | 0.125° | Daily | m |
| **Ch 3 (Mask 10)** | Surface Zonal Current (U) | Copernicus Multi-Obs Drifter+Altimetry | 0.25° | Daily | m/s |
| **Ch 4 (Mask 11)** | Surface Meridional Current (V) | Copernicus Multi-Obs Drifter+Altimetry | 0.25° | Daily | m/s |
| **Ch 5 (Mask 12)** | 10m Neutral Zonal Wind (U) | Copernicus Scatterometer L4 (`012_006`) | 0.125° | Hourly -> Daily Mean | m/s |
| **Ch 6 (Mask 13)** | 10m Neutral Meridional Wind (V) | Copernicus Scatterometer L4 (`012_006`) | 0.125° | Hourly -> Daily Mean | m/s |

### 3. Binary Quality Masking Protocol
To prevent convolutional filters from diffusing NaN values across marine coastlines, physical channels 0–6 substitute missing/land pixels with **0.0**, accompanied by explicit binary mask channels 7–13 where:
M_c(x, y) = 1.0 if authentic ocean observation exists, and 0.0 if landmass, bathymetric barrier, or missing data.
The network's first convolution learns to weight inputs proportionally to their validity flags.

### 4. Subsurface Reference Target (GLORYS12V1)
- **Dataset:** Copernicus `GLOBAL_MULTIYEAR_PHY_001_030` (Mercator Ocean GLORYS12V1).
- **Variable:** 3D Potential Temperature (`thetao`, units: °C).
- **Vertical Interpolation:** 1D linear interpolation across 50 native levels bounded within [0.494 m, 1062.5 m]; surface (0 m) extrapolated from 0.494 m.
- **Horizontal Regridding:** Bilinear interpolation from native 0.083° to 0.25° target grid with strict retention of coastal/bathymetric masks.

---

## Page 3: Deep Neural Architecture (OceanEmbedNetV3_Decoder)

### 1. Structural Breakthrough: Parallel Multi-Depth Projection
In Phase 4, the initial OceanEmbedNet evaluated each depth sequentially by passing scalar depth indices through a shared decoder in an explicit loop, taking ~172.4 s/epoch.  
**OceanEmbedNetV3_Decoder** restructures the architecture into a unified multi-scale U-Net with a **parallel multi-depth vertical projection head** (`nn.Conv2d(32, 15, kernel_size=1)`), predicting all 15 depths simultaneously in a single forward pass. This achieved a **7.2× training throughput acceleration (~23.9 s/epoch)**.

### 2. Layer & Dimension Breakdown

| Stage | Submodule | Operation | Input Shape | Output Shape | Parameters |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Encoder** | `enc1` | DoubleConv(14, 32) | $[B, 14, 101, 241]$ | $[B, 32, 101, 241]$ | 13,568 |
| | `pool1` | MaxPool2d(2) | $[B, 32, 101, 241]$ | $[B, 32, 50, 120]$ | 0 |
| | `enc2` | DoubleConv(32, 64) | $[B, 32, 50, 120]$ | $[B, 64, 50, 120]$ | 55,616 |
| | `pool2` | MaxPool2d(2) | $[B, 64, 50, 120]$ | $[B, 64, 25, 60]$ | 0 |
| | `enc3` | DoubleConv(64, 128) | $[B, 64, 25, 60]$ | $[B, 128, 25, 60]$ | 221,824 |
| | `pool3` | MaxPool2d(2) | $[B, 128, 25, 60]$ | $[B, 128, 12, 30]$ | 0 |
| **Bottleneck** | `bottleneck` | DoubleConv(128, 128) | $[B, 128, 12, 30]$ | $[B, 128, 12, 30]$ | 295,296 |
| **Decoder** | `up3` + `dec3` | ConvTranspose2d + Cat + DoubleConv | $[B, 128, 12, 30]$ | $[B, 128, 25, 60]$ | 508,288 |
| | `up2` + `dec2` | ConvTranspose2d + Cat + DoubleConv | $[B, 128, 25, 60]$ | $[B, 64, 50, 120]$ | 139,456 |
| | `up1` + `dec1` | ConvTranspose2d + Cat + DoubleConv | $[B, 64, 50, 120]$ | $[B, 32, 101, 241]$ | 41,408 |
| **Projection** | `final_conv` | Conv2d(32, 15, kernel_size=1) | $[B, 32, 101, 241]$ | $[B, 15, 101, 241]$ | 495 |
| **Prior** | `climatology_prior` | Learnable parameter vector | — | $[1, 15, 1, 1]$ | 15 |
| **TOTAL** | — | **OceanEmbedNetV3_Decoder** | — | — | **1,275,934** |

- **Total Trainable Parameters:** **1,275,934**
- **Total State Dict Elements:** **1,278,252** (includes 2,318 BatchNorm running mean/variance buffers).
- **Physical Inductive Bias:** The climatological prior vector is initialized with North Indian Ocean regional stratification: `[28.2, 28.1, 27.9, 27.5, 26.8, 25.0, 22.5, 19.8, 17.2, 15.1, 13.0, 10.5, 8.4, 7.0, 5.2] °C`. `final_conv` bias is initialized to zero so the network learns residual anomaly departures.

---

## Page 4: Training, Validation & Model Selection

### 1. Strict Temporal Splitting Protocol
To eliminate future-period leakage and simulate real operational prediction, the 274 daily fields are partitioned chronologically:
- **Training Set (Jan 1 – Jul 31, 2020):** 213 daily fields (`chunk_2020_01.pt` to `chunk_2020_07.pt`).
- **Validation Set / Model Selection Gate (Aug 1 – Aug 31, 2020):** 31 daily fields (`chunk_2020_08.pt`).
- **Completely Held-Out Test Set (Sep 1 – Sep 30, 2020):** 30 daily fields (`chunk_2020_09.pt`).

### 2. Zero-Leakage Normalization
The standard scaler parameters (channel means, standard deviations) were fit **strictly on January–July 2020 training data** (N = 213). Validation and test fields did not contribute to scaler statistics.

### 3. Model Selection Gate (August 2020 Validation)
In Phase 5, all candidate models were compared on August 2020 validation RMSE:
- Original OceanEmbedNet: 1.8867°C
- Pointwise MLP: 1.0836°C
- Simple CNN Baseline: 0.9739°C
- EXP-01 (Thermocline-Aware Loss): 1.8978°C
- EXP-04 (Residual Anomaly Network): 0.8433°C
- **EXP-02 (OceanEmbedNetV3_Decoder):** **0.7570°C (Selected Winner)**

*Critical Scientific Note:* September GLORYS and ARGO data were **never used during model selection**.

### 4. Held-Out September 2020 GLORYS Reference Results (4,601,790 Valid Cells)

| Model Architecture | Parameters | GLORYS RMSE | GLORYS MAE | Pearson r | Error vs Simple CNN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Static Regional Climatology | 15 (prior) | 2.9976°C | 2.4705°C | 0.9663 | Baseline Reference |
| Pointwise MLP Baseline | 34,831 | 1.1177°C | 0.7569°C | 0.9893 | +7.3% error |
| Simple CNN Baseline | 1,210,639 | 1.0418°C | 0.7100°C | 0.9907 | Standard Baseline |
| Original OceanEmbedNet (Phase 4) | 1,342,928 | 1.8142°C | 1.3229°C | 0.9741 | +74.1% error |
| **OceanEmbedNetV3_Decoder (Ours)** | **1,275,934** | **0.8601°C** | **0.5780°C** | **0.9936** | **-17.44% error** |

### 5. Independent In-Situ ARGO Observational Check
- **Data Source:** Coriolis / INCOIS Global Data Assembly Centre (September 1, 15, 25, 2020).
- **Sample Composition:** Exactly **497 matched profile-depth observations from 36 authentic ARGO profiles across 28 unique WMO floats** (14 depths, 5m to 1000m; 0m has no ARGO sensor measurements).
- **Overall ARGO RMSE:**
  - Climatology: 3.0363°C
  - Original OceanEmbedNet: 1.8126°C
  - Simple CNN Baseline: 0.9526°C
  - **OceanEmbedNetV3_Decoder:** **0.7973°C (-16.30% error reduction vs CNN)**

---

## Page 5: Depth-Wise Performance & Real System Integration

### 1. Depth-Wise ARGO RMSE Evaluation

| Depth Level | Ocean Regime | Simple CNN RMSE | OceanEmbed v3 RMSE | Error Reduction (Delta) | Relative Gain | Superiority |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **5 m** | Near-Surface | 0.5825°C | **0.5284°C** | -0.0541°C | +9.3% | **PASS** |
| **10 m** | Mixed Layer | 0.7618°C | **0.6030°C** | -0.1588°C | +20.8% | **PASS** |
| **20 m** | Mixed Layer | 0.8695°C | **0.6582°C** | -0.2113°C | +24.3% | **PASS** |
| **30 m** | Mixed Layer | 0.9604°C | **0.7194°C** | -0.2410°C | +25.1% | **PASS** |
| **50 m** | Upper Thermocline | 1.0366°C | **0.8679°C** | -0.1687°C | +16.3% | **PASS** |
| **75 m** | **Main Thermocline** | 1.2220°C | **1.0388°C** | -0.1832°C | +15.0% | **PASS** |
| **100 m** | **Peak Thermocline** | 1.6876°C | **1.5317°C** | -0.1559°C | +9.2% | **PASS** |
| **125 m** | **Main Thermocline** | 1.1984°C | **1.0631°C** | -0.1353°C | +11.3% | **PASS** |
| **150 m** | **Lower Thermocline** | 1.0278°C | **0.8645°C** | -0.1633°C | +15.9% | **PASS** |
| **200 m** | Intermediate | 1.1067°C | **0.9100°C** | -0.1967°C | +17.8% | **PASS** |
| **300 m** | Intermediate | 0.6147°C | **0.3585°C** | -0.2562°C | +41.7% | **PASS** |
| **500 m** | Deep Ocean | 0.4532°C | **0.2997°C** | -0.1535°C | +33.9% | **PASS** |
| **700 m** | Deep Ocean | 0.3757°C | **0.2631°C** | -0.1126°C | +30.0% | **PASS** |
| **1000 m** | Deep Ocean | 0.4037°C | **0.2586°C** | -0.1451°C | +35.9% | **PASS** |

- **Key Finding:** OceanEmbed v3 achieves lower ARGO RMSE than Simple CNN at **all 14 evaluated depths**.
- **Thermocline Dynamics (75–150m):** High gradients (>0.1°C/m) produce the largest absolute errors, yet OceanEmbed v3 achieves consistent 0.14°C to 0.18°C reductions.
- **Deep Stability (500–1000m):** Reaches **0.2586°C RMSE at 1000m** (35.9% lower than CNN).

### 2. Real Live Inference Demonstration (15.00°N, 85.00°E — Central Bay of Bengal)
- **Snapshot Date:** 2020-09-15
- **Extracted Surface Observations:**
  - SST: 29.547°C | SSS: 33.397 psu | SSH: 0.200 m
  - Zonal Current: -0.051 m/s | Meridional Current: 0.168 m/s
  - Zonal Wind: 5.017 m/s | Meridional Wind: 6.696 m/s
- **Reconstructed 15-Depth Vertical Column:**
  - 0m: 29.65°C, 5m: 29.59°C, 10m: 29.56°C, 20m: 29.61°C, 30m: 29.53°C
  - 50m: 28.94°C, 75m: 26.82°C, 100m: 23.75°C, 125m: 20.60°C, 150m: 17.92°C
  - 200m: 14.68°C, 300m: 11.90°C, 500m: 10.04°C, 700m: 8.58°C, 1000m: 6.68°C
- **Derived Physical Indicators:**
  - Mixed Layer Depth (MLD, 0.2°C threshold): **32.88 m**
  - Thermocline Depth (max |dT/dz|): **112.50 m**
  - Upper Ocean Heat Content (OHC 0–300m): **24.465 GJ/m²**

### 3. Backend Architecture & Latency Benchmark
- **Framework:** FastAPI with a singleton service architecture. Model and scaler loaded once on startup.
- **Inference Mode:** `torch.inference_mode()` on CPU.
- **Audited Latency Profile (30 iterations):**
  - Pure Neural Model Forward: **31.85 ms**
  - Preprocessing + Coordinate Mapping + JSON Serialization: **4.03 ms**
  - Total Roundtrip API Latency: **Mean: 35.87 ms | Median: 34.92 ms | p95: 44.84 ms**

---

## Page 6: Demonstration, Scientific Boundaries & Jury Defense

### 1. Live Demonstration Workflow
1. **User Interaction:** Operator selects coordinates on the Leaflet map or inputs custom coordinates within 5°N–30°N, 45°E–105°E and chooses an observation date from the 2020 archive (274 available dates).
2. **API Request:** React frontend dispatches JSON request to `POST /predict`.
3. **Zero-Leakage Preprocessing:** Backend validates coordinates, checks ocean validity (Channel 7 mask), extracts the 14-channel input field, and normalizes using the frozen training scaler.
4. **Neural Forward Pass:** OceanEmbed v3 predicts the 15-depth temperature profile in ~35 ms.
5. **Dynamic Dashboard Rendering:** Frontend displays the SVG temperature profile, highlights the thermocline inflection, and calculates MLD and OHC300.

### 2. Scientific Boundaries & Explicit Limitations
- **Archive Demonstration:** Operates on a reprocessed 2020 archive (274 daily fields). **This is a research prototype, not a real-time operational satellite feed.**
- **Reanalysis Reference Target:** GLORYS12V1 is used as a numerical reanalysis reference target, rather than a direct in-situ observational sensor measurement.
- **Observational Sample Scope:** ARGO verification is based on selected September 2020 snapshots (497 matched profile-depth pairs across 28 floats).
- **Temporal Generalization:** Extending reconstruction across different years and extreme climate modes (positive/negative IOD, El Niño) remains future validation scope.
- **Domain Restriction:** Predictions are strictly constrained to the North Indian Ocean basin (5°N–30°N, 45°E–105°E) down to 1000 m depth.

### 3. Anticipated Jury Defense Guide
- **Q1: Did you train on ARGO or select models using ARGO?**  
  *Answer:* Absolutely not. ARGO observations were completely untouched throughout model development and hyperparameter tuning. OceanEmbed v3 was selected based strictly on August 2020 GLORYS validation. ARGO was used solely as a blind observational reality check.
- **Q2: Why does error peak around 100m depth?**  
  *Answer:* The 75–150m layer represents the tropical permanent thermocline, where vertical temperature gradients exceed 0.1°C/m and internal wave displacements induce high temporal variance. Despite this physical difficulty, OceanEmbed v3 reduced thermocline error by up to 0.18°C compared to the baseline CNN.
- **Q3: Can deep learning replace physical numerical models like NEMO?**  
  *Answer:* No. OceanEmbed serves as an ultra-fast statistical reconstruction emulator (~35 ms) complementary to dynamical assimilation systems.

### 4. Master Project Takeaway
> *"OceanEmbed demonstrates the feasibility of reconstructing subsurface ocean temperature from surface satellite observations using a compact deep-learning architecture with independent observational checking."*
