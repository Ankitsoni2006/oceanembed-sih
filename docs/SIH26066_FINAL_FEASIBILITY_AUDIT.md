# SIH26066 — COMPLETE DATA FEASIBILITY AND INTEGRATION AUDIT (FINAL REVISION)

**Project:** SIH26066 — Deep Learning Subsurface Ocean Temperature Profile Reconstruction  
**Audit Type:** Empirical, Code-Level, and Real NetCDF Integration Audit  
**Date:** September 2026  
**Auditor:** Antigravity Autonomous Pair Programmer  
**Final Decision:** **GO**

---

## 1. SSS VERIFICATION & CHANNEL REMAPPING

In the initial integration script, Channel 1 (SSS) temporarily utilized surface salinity extracted from the GLORYS pilot. Per audit requirements, this was disqualified under Rule 6 ("Do not use GLORYS surface variables as final satellite inputs"). 

A dedicated pilot download was executed for the official Copernicus Multi-Obs Sea Surface Salinity product:
- **Dataset ID:** `cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c`
- **File Downloaded:** `data/pilot/sss_pilot.nc` ($488,082\text{ bytes}$)
- **Time Window:** 2019-12-25 to 2020-01-08 (bounding 2020-01-01)
- **Time Processing:** Linear interpolation between weekly granules to `2020-01-01T00:00:00`.
- **Spatial Processing:** Bilinear interpolation to the $0.25^\circ$ NIO grid ($101 \times 241$).
- **Saved Tensor Pair:** `data/pilot/sample_X_Y_real.pt` ($2,825,445\text{ bytes}$).

### Exact Physical Channel Mapping & Channel Statistics
Evaluated directly on the saved tensor $X$ ($101 \times 241 = 24,341\text{ total grid cells}$):

| Channel | Variable | Underlying Source | Units | Min | Max | Mean | Std | NaN Count (in raw) | Valid Fraction |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ch 0** | **SST** | OSTIA L4 Reprocessed | °C | 20.105 | 31.267 | 28.890 | 1.534 | 12,379 | **49.14%** |
| **Ch 1** | **SSS** | Multi-Obs L4 Satellite Salinity | psu | 27.856 | 44.128 | 34.568 | 1.954 | 14,099 | **42.08%** |
| **Ch 2** | **SSH / SLA** | DUACS L4 Altimetry | m | -0.302 | 0.355 | 0.067 | 0.094 | 12,538 | **48.49%** |
| **Ch 3** | **Surface U Current** | Multi-Obs L4 Currents | m/s | -0.998 | 0.662 | -0.103 | 0.188 | 13,069 | **46.31%** |
| **Ch 4** | **Surface V Current** | Multi-Obs L4 Currents | m/s | -0.894 | 0.772 | 0.037 | 0.184 | 13,069 | **46.31%** |
| **Ch 5** | **Surface U Wind** | Scatterometer L4 Winds | m/s | -10.879 | 5.950 | -2.491 | 2.657 | 680 | **97.21%** |
| **Ch 6** | **Surface V Wind** | Scatterometer L4 Winds | m/s | -8.852 | 9.083 | -1.060 | 2.117 | 680 | **97.21%** |

*Channels 7 through 13 contain the corresponding binary validity masks ($1.0$ for observed ocean, $0.0$ for land/missing).*

---

## 2. COMMON TIME PERIOD (EMPIRICALLY VERIFIED)

Analyzed across the exact bounding timestamps of all 8 required datasets:
- **GLORYS Reanalysis (`cmems_mod_glo_phy_my_0.083deg_P1D-m`):** 1993-01-01 to 2026-06-23
- **SST (`METOFFICE-GLO-SST-L4-REP-OBS-SST`):** 1981-08-24 to 2026-03-31
- **SSS (`cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c`):** 2010-06-03 to 2025-12-25
- **SSH / SLA (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`):** 1993-01-01 to 2026-03-31
- **U/V Current (`cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m`):** 1993-01-01 to 2026-03-31
- **U/V Wind (`cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H`):** 2007-01-11 to 2026-04-21

### Findings:
- **Earliest Common Date:** **2010-06-03**
- **Latest Common Date:** **2025-12-25**
- **Limiting Dataset:** **SSS (Multi-Obs Satellite Salinity)** determines both the earliest and latest valid dates.
- **Total Valid Multi-Year Overlap:** **5,684 days (15.6 years)**.

---

## 3. REAL X / Y TENSOR PROOF

- **$X$ Tensor Shape:** `torch.Size([1, 14, 101, 241])` ($1.30\text{ MB}$, float32)
- **$Y$ Tensor Shape:** `torch.Size([1, 15, 101, 241])` ($1.39\text{ MB}$, float32)
- Target $Y$ corresponds to the 15 SIH target depths: $0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{ m}$.
- All 14 input channels and 15 target channels are sourced from genuine downloaded NetCDFs.

---

## 4. LAND MASK & LOSS CONTRIBUTION AUDIT

- **Total Grid Cells:** $24,341$ ($101 \times 241$)
- **Ocean Cells:** $11,854$
- **Land Cells:** $12,487$
- **Ocean Percentage:** **$48.70\%$**

### Mathematical Loss Isolation Proof:
In `scripts/rebuild_real_xy_with_genuine_sss.py`, an adversarial test evaluated two predictions:
1. Standard prediction over valid ocean pixels.
2. Prediction where all $12,487$ land pixels were corrupted with extreme values ($999,999.0$).
- **Loss 1 (Clean Ocean):** $493.4060$
- **Loss 2 (Corrupted Land):** $493.4060$
- **Delta:** $0.000000$
- **Proof:** Land pixels contribute strictly $0.000$ to the loss function and cannot leak into gradients.

---

## 5. ACTUAL MEASURED COMPUTE BENCHMARKS (NO EXTRAPOLATIONS)

Measured on local CPU machine using `scripts/benchmark_compute.py` ($B=4$):
- **Input Tensor Size ($B=4$):** **$5.20\text{ MB}$**
- **Output Tensor Size ($B=4$):** **$5.57\text{ MB}$**
- **Base Process RAM:** **$194.54\text{ MB}$**
- **Peak Process RAM:** **$1,661.93\text{ MB}$ ($1.66\text{ GB}$)**
- **Mean Forward Pass Time:** **$1,737.74\text{ ms}$**
- **Mean Backward Pass Time:** **$3,383.27\text{ ms}$**
- **Total Measured Training Step Time:** **$5,133.83\text{ ms}$ per batch of 4** ($0.78\text{ ocean-days / second}$ on CPU)
- *Note:* Full-year training duration extrapolations have been removed. Only directly measured hardware benchmarks are reported.

---

## 6. TERMINOLOGY CLARIFICATION & REAL-DATA TRAINING STEP

The term "model convergence" is strictly retracted and replaced with **"real-data training-step integration test"**. Multi-epoch convergence was not tested on this single-day pilot, nor was it intended.
- **Single-Sample Real Training Step Latency:** **$1,409.68\text{ ms}$**
- **Step Loss:** $509.06$ (Masked MSE over un-normalized physical units; serves as an autograd connectivity verification, not a performance metric).
- **Autograd Health:** Active finite gradients verified across all encoder, decoder, bottleneck, and embedding parameters.

---

## 7. ARGO VALIDATION CLARIFICATION

- **INCOIS Web Portal Access:** **UNVERIFIED / 404**. The INCOIS LAS web endpoint returned HTTP 404 due to portal restructuring.
- **Independent In-Situ ARGO Validation Pipeline:** **EXPERIMENTALLY VERIFIED**. Sourced from the official Global Data Assembly Centre (Coriolis GDAC, HTTP 200, `https://data-argo.ifremer.fr/geo/indian_ocean/`).
  - Real NetCDF file downloaded: `data/argo/20221101_prof.nc` ($6.14\text{ MB}$).
  - Float #34 ($17.9540^\circ\text{N}, 63.5540^\circ\text{E}$) successfully colocated to $(18.00^\circ\text{N}, 63.50^\circ\text{E})$ and vertically interpolated across all 15 SIH depths.

---

## 8. FINAL DECISION

# **GO**

All 7 required surface inputs (including genuine satellite SSS), the 3D GLORYS target, spatial regridding, the PyTorch training step, and independent ARGO colocation are experimentally verified on real data.
