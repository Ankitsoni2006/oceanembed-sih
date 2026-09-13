# SIH26066 — Phase 4 Stage 9 Independent In-Situ ARGO Validation Report

## 1. Scientific Distinction & Verification Protocol

> **CRITICAL SCIENTIFIC DISTINCTION & TEMPORAL DISCLOSURE**:  
> - **GLORYS12V1**: Reanalysis numerical model target used for supervised training and historical benchmark comparison. Not direct observation.  
> - **ARGO Autonomous Floats**: True unassimilated in-situ physical profiling measurements collected by robotic CTD profilers in the ocean water column. True independent observational validation.
> - **TEMPORAL GAP DISCLOSURE (AUDIT CHECK 12)**: The local Coriolis float profile archive (`data/argo/20221101_prof.nc`) records observations on **2022-11-01**, whereas the satellite test inputs represent **September 2020** (a **762-day gap**). This evaluation is classified as a **Cross-Temporal Climatological Structure Transfer Test**, testing whether the network captures physical stratification profiles that generalize across years. It **must NOT be claimed as contemporaneous validation** of the September 2020 synoptic ocean state.

- **ARGO Archive**: Coriolis / INCOIS Global Data Assembly Centre (`data/argo/20221101_prof.nc`)  
- **Matched Float Profiles**: 8 autonomous profiling floats strictly within the NIO basin (5°N–30°N, 45°E–105°E)  
- **Vertical Colocation**: Linear interpolation strictly within sensor bounds; zero unphysical deep extrapolation  
- **Scientific Audit Status**: `WARNING` — Valid as cross-temporal structure transfer; invalid for contemporaneous validation.  

## 2. Independent ARGO Observational Performance Comparison

| Model | Total Obs | ARGO RMSE (°C) | ARGO MAE (°C) | ARGO Bias (°C) | Pearson r |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Climatology Profile** | 107 | **2.7911** | 2.2475 | -1.8583 | 0.9668 |
| **Pointwise MLP** | 107 | **2.1666** | 1.5151 | +0.4315 | 0.9539 |
| **Simple CNN Baseline** | 107 | **2.0043** | 1.3811 | +0.3873 | 0.9598 |
| **OceanEmbedNet (Multi-Scale U-Net)** | 107 | **1.7679** | 1.3393 | -0.2728 | 0.9695 |

## 3. Depth-Stratified ARGO Evaluation

| Depth | Observations | Climatology RMSE | PointwiseMLP RMSE | SimpleCNN RMSE | OceanEmbed RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0 m** | 0 | N/A | N/A | N/A | N/A |
| **5 m** | 7 | 0.7447°C | 1.3011°C | 1.2028°C | **0.9709°C** |
| **10 m** | 8 | 0.8450°C | 1.1062°C | 1.0795°C | **0.8306°C** |
| **20 m** | 8 | 1.0065°C | 1.1506°C | 1.1612°C | **0.9169°C** |
| **30 m** | 8 | 1.5025°C | 1.9710°C | 1.8880°C | **1.3443°C** |
| **50 m** | 8 | 1.9768°C | 2.4208°C | 2.2126°C | **1.9341°C** |
| **75 m** | 8 | 2.5997°C | 3.5924°C | 3.4308°C | **2.9256°C** |
| **100 m** | 8 | 2.2689°C | 3.4555°C | 3.3162°C | **2.3787°C** |
| **125 m** | 8 | 2.6173°C | 2.7969°C | 2.7321°C | **2.2216°C** |
| **150 m** | 8 | 3.4646°C | 2.2511°C | 2.1253°C | **2.1918°C** |
| **200 m** | 8 | 4.1787°C | 2.1238°C | 1.7912°C | **2.1440°C** |
| **300 m** | 8 | 4.1594°C | 1.7816°C | 1.2591°C | **1.7067°C** |
| **500 m** | 8 | 4.0163°C | 1.6553°C | 1.2078°C | **1.0928°C** |
| **700 m** | 7 | 3.3442°C | 0.8492°C | 0.5996°C | **0.6431°C** |
| **1000 m** | 5 | 2.6471°C | 0.5894°C | 0.5280°C | **1.1856°C** |

