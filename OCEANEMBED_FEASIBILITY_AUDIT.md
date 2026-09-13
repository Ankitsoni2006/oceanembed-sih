---
title: "MISSION: FINAL GO / NO-GO FEASIBILITY AUDIT"
author: SIH 2026 Team
pdf_options:
  format: a4
  margin: 20mm
  printBackground: true
---

# SIH26066 — OCEANEMBED FINAL FEASIBILITY AUDIT

## 1. EXACT SIH REQUIREMENTS
**[Official Source: SIH 26066 Problem Statement]**
*   **Region:** North Indian Ocean (5°N–30°N, 45°E–105°E)
*   **Output Spatial/Temporal:** 0.25° × 0.25° / Daily
*   **Surface Inputs:** SST, SSS, SSH/SLA, Surface U current, Surface V current, Wind U, Wind V
*   **Required Target Depths:** 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
*   **Training Target:** GLORYS Global Ocean Reanalysis
*   **Independent Validation:** Gridded ARGO / INCOIS LAS
*   **Expected Metrics:** RMSE, Correlation, Bias

## 2. CRITICAL DATA FEASIBILITY AUDIT
Investigation of the current Copernicus Marine and standard catalogs confirms the availability of required products:
*   **GLORYS 3D Temperature:** `cmems_mod_glo_phy_my_0.083deg_P1D-m` (GLORYS12V1). Daily, 1/12°, covers 1993–onward.
*   **SST:** Copernicus `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011` (OSTIA) or similar L4 products. Daily, 0.05°.
*   **SSS / SSH / Currents:** Multi-observation L4 products or extracted directly from surface levels of GLORYS if observation-only products are sparse.
*   **Winds (U/V):** Copernicus `WIND_GLO_WIND_L4_REP_OBSERVATIONS_012_006` or ERA5 (ECMWF) daily aggregates. 0.25°.
*   **Feasibility:** All datasets are available, permit historical downloading, allow spatial subsetting, and support daily intervals.

## 3. GLORYS 3D DATA — HARD FEASIBILITY TEST
*   **Product:** `GLOBAL_MULTIYEAR_PHY_001_030` (`cmems_mod_glo_phy_my_0.083deg_P1D-m`)
*   **Target Variable:** `thetao` (Potential Temperature).
*   **Verification:** The dataset contains 50 native vertical depth levels reaching 5500m. It is a true 3D product, not a surface slice. 
*   **Subset capability:** The Copernicus Motu client / API supports spatial (5-30N, 45-105E), temporal, and depth subsetting prior to download.
*   *(Note: Local programmatic download pilot was bypassed as the workspace Copernicus credentials were reset. The API metadata conclusively confirms 3D variable presence).*

## 4. NATIVE DEPTH → SIH DEPTH FEASIBILITY
*   **GLORYS Native Levels:** 50 standard levels. 22 levels reside in the top 100m (approx 1m resolution near-surface), expanding safely through 1000m to 5500m.
*   **Interpolation Safety:** Because the native vertical grid brackets and exceeds all 15 SIH required depths (0 to 1000m) with high density in the upper ocean, it is scientifically safe and standard practice to vertically interpolate native `thetao` locally to the 15 exact SIH levels.

## 5. SURFACE DATA HARMONIZATION
*   **Requirement:** Harmonize to 0.25° × 0.25°.
*   **Region Size:** 25° Lat (100 pixels) × 60° Lon (240 pixels) = **24,000 pixels per day**.
*   **Data Volume:** 24,000 pixels × 7 channels × 4 bytes ≈ ~670 KB per day for inputs.
*   **Storage:** 1 year ≈ 245 MB. 10 years ≈ 2.5 GB. 
*   **Feasibility:** Extremely feasible. Harmonization via xarray/scipy (bilinear/nearest) onto a common 0.25° grid is computationally trivial.

## 6. COMMON TEMPORAL WINDOW
A conservative, scientifically defensible common overlap where GLORYS, modern Satellite L4 products, and mature ARGO data coexist is the post-2010 era. 
*   **Recommended Split:**
    *   **Training:** 2012–2020
    *   **Validation:** 2021
    *   **Test:** 2022
*   This avoids the sparsity of early-2000s ARGO and aligns with high-quality satellite eras.

## 7. ARGO VALIDATION FEASIBILITY
*   **Source:** INCOIS Live Access Server (LAS).
*   **Product:** INCOIS provides Gridded ARGO Sea Surface/Subsurface products (DIVA method, 0.25° × 0.25°) and raw profile data.
*   **Feasibility:** Highly practical. Matching does not require exact point equality; validation can be performed by gridding ARGO floats within the month/week to the 0.25° grid or comparing model outputs directly at float locations using a spatial/temporal tolerance window.

## 8. EXISTING DEEP-LEARNING IMPLEMENTATIONS
Extensive peer-reviewed literature confirms the technical viability of the task:
1.  **DORS (Deep Ocean Remote Sensing):** Uses ConvLSTM on SST/SSH/SSS to predict down to 2000m. (Lu et al.)
2.  **U-Net/CNN Variants:** VI-UNet and 3D U-Net++ have been explicitly published for estimating 3D ocean temperature from surface observations.
*   **Conclusion:** The modeling task (mapping surface to subsurface) is proven executable. A team of six students can realistically reproduce a streamlined U-Net architecture.

## 9. OUR PROPOSED SOLUTION (OceanEmbed)
*   **Architecture:** Masked Multi-Scale U-Net → Latent Ocean Embedding → Depth-Conditioned Profile Decoder.
*   **Feasibility:** 
    *   *Implementation:* High. Standard PyTorch components (Conv2D, Linear).
    *   *Compute:* A compact U-Net processes a 100x240 spatial grid easily on consumer GPUs.
    *   *Risk:* Low. We aren't inventing new math; we are applying robust spatial architectures to a localized grid.

## 10. BASELINE-FIRST FEASIBILITY
The project only works if we start simple. 
*   **Level 0:** Climatology (Monthly means).
*   **Level 1:** Point-wise MLP (1x1 pixel mapping).
*   **Level 3:** Plain CNN.
*   **Level 6 (Target):** OceanEmbed (U-Net + Embedding + Depth Decoder).
*   **Feasibility:** Training Level 1 (MLP) on 24,000 pixels/day takes minutes. This guarantees a working PoC baseline.

## 11. COMPUTE FEASIBILITY
*   **Grid:** 100 × 240. 
*   **VRAM:** A batch size of 16 for a lightweight U-Net on a 100x240 image requires < 4GB VRAM.
*   **Hardware:** Can easily be trained on Google Colab (T4 16GB) or an 8GB consumer NVIDIA GPU. 
*   **Time:** A 10-year dataset (~3,650 samples) will train in under 2-3 hours on a T4 GPU.

## 12. TEAM OF SIX FEASIBILITY
*   **Member 1:** Data acquisition (Copernicus API/Motu).
*   **Member 2:** Harmonization (xarray regridding, QC).
*   **Member 3:** Baselines (MLP, Climatology).
*   **Member 4:** OceanEmbed Architecture (PyTorch).
*   **Member 5:** ARGO Validation pipeline.
*   **Member 6:** Integration, Experiments, Docs.
*   **Risk:** Member 1/2 are the primary bottlenecks. If data isn't prepared, 3 and 4 cannot work.

## 13. DEVELOPMENT TIMELINE
*   **PHASE 0:** Data access proof (72 hrs) - *Dependency: Motu API Auth.*
*   **PHASE 1:** 7-day pilot (1 week) - *Deliverable: 1 NetCDF file paired.*
*   **PHASE 2:** 1-month paired dataset (1 week) - *Go/No-Go for full training.*
*   **PHASE 3:** Baselines (1 week).
*   **PHASE 4 & 5:** OceanEmbed (2 weeks).
*   **PHASE 6:** ARGO validation (1 week).

## 14. IDENTIFY EVERY REAL BLOCKER
| BLOCKER | SEVERITY | PROBABILITY | CURRENT STATUS | MITIGATION |
| :--- | :--- | :--- | :--- | :--- |
| **Copernicus Authentication** | HARD | Low | Password reset | Fix API keys/auth immediately. |
| **GLORYS 3D Size limit** | Medium | High | API caps | Download in yearly/monthly chunks. |
| **Missing Clouds/Data** | Medium | High | Known satellite issue | Use L4 gap-free products & Validity Masks. |
| **1000m accuracy loss** | Research | High | Expected physics decay | Focus evaluation on relative improvement over climatology. |

## 15. 1000 m RISK
Is it scientifically plausible to reconstruct 1000m? 
Yes, but with caveats. Surface manifestations (SSH altimetry) reflect integrated water column density, allowing models to infer thermocline depth and deep structures. However, skill (Correlation) naturally degrades and RMSE tightens (as deep variance is low) below 500m. 
*   **Evaluation:** Depth-wise skill is a primary evaluation axis. We must not pretend 1000m will have the exact same dynamic tracking accuracy as 50m.

## 16. WHAT CAN WE GUARANTEE VS WHAT CANNOT?
**WE CAN BE CONFIDENT ABOUT:**
*   Required data products exist and are accessible.
*   The GLORYS product provides 50 native depths bracketing 1000m.
*   A 100x240 grid is computationally trivial for deep learning.
*   Deep learning spatial reconstruction has proven precedent.

**WE CANNOT GUARANTEE:**
*   Exact RMSE/Correlation scores against ARGO.
*   Whether 1000m prediction outperforms simple climatology significantly.
*   Whether advanced temporal extensions (ConvLSTM) will justify their compute cost over spatial-only OceanEmbed.

## 17. DEFINE THE MINIMUM PROOF REQUIRED TO COMMIT
Do not commit fully until completing the **72-Hour Commitment Gates**:
1.  **TEST 1:** Successfully download 7 days of real 3D GLORYS `thetao` data (5-30N, 45-105E).
2.  **TEST 2:** Download corresponding 7 days of surface SST and SSH.
3.  **TEST 3:** Interpolate GLORYS to the 15 SIH depths.
If these API/Data extraction steps fail, the project is dead.

## 18. DO NOT RECOMMEND SWITCHING PS TOO EARLY
The PS is highly feasible computationally and theoretically. The *only* genuine hard blockers would be a permanent inability to authenticate with Copernicus or the INCOIS server going permanently offline. Algorithmic difficulty or poor initial RMSE are *not* reasons to abandon; they are expected research challenges.

---
# 19. SIH26066 FINAL FEASIBILITY VERDICT

**VERDICT: 🟡 CONDITIONAL GO**

*   **DATA FEASIBILITY:** 8/10
*   **MODEL FEASIBILITY:** 9/10
*   **COMPUTE FEASIBILITY:** 10/10
*   **VALIDATION FEASIBILITY:** 8/10
*   **TEAM EXECUTION FEASIBILITY:** 8/10
*   **RESEARCH RISK:** 3/10 (Lower is better)

### Why we should commit
1.  **Compute is Trivial:** A 100x240 grid allows for incredibly fast iteration on consumer GPUs.
2.  **Data Exists:** GLORYS12V1 officially provides the 50-level 3D data required.
3.  **Proven Concept:** DORS and U-Net variants are already published proving the surface-to-subsurface mapping works.
4.  **Distinct Architecture:** Our OceanEmbed framing (Masked U-Net + Depth Decoder) gives us a concrete, defensible engineering narrative for judges.

### What could still kill the project
1.  Permanent failure to authenticate and automate Copernicus Motu/API downloads.
2.  Data engineering bottlenecks if Members 1 & 2 cannot quickly build the harmonization pipeline.

### What must be completed in the next 48–72 hours
1.  Fix Copernicus Marine credentials.
2.  Write a Python script to download 7 days of 3D `thetao` for the NIO region.
3.  Verify the downloaded NetCDF actually contains `depth` dimensions spanning 0 to 1000m.

### Final recommendation
**COMMIT TO PS66 ONLY AFTER THE 72-HOUR DATA GATES ARE PASSED.**

---
**The Most Important Question:**
*“If six technically capable B.Tech students started working on SIH26066 tomorrow, do they have a realistic path to producing a working, scientifically defensible PoC before the hackathon?”*

**Answer:** Yes, absolutely. The computational scale of the NIO region (100x240 grid) makes this entirely feasible on free cloud tiers (Colab) or student laptops. The primary barrier is not AI math; it is purely data wrangling (NetCDF formatting, Regridding). If the team can download and align the NetCDF files in week one, the deep learning implementation will succeed.
