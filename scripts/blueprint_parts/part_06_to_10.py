# -*- coding: utf-8 -*-
"""
SIH26066 Master Decision Guide - Parts 6 to 10
"""

def get_parts():
    p6 = """# PART 6: WHY THIS WAS NOT A RANDOM CHOICE (DECISION FRAMEWORK)

Our team utilized a multi-criteria weighted scoring framework to evaluate competing SIH problem statements before committing to SIH26066.

### 1. Decision Evaluation Matrix

| Decision Criterion | Weight | Score (1-10) | Weighted Score | Justification & SIH26066 Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Scientific & Technical Depth** | 20% | 9.5 | 1.90 | Combines satellite remote sensing, geophysical fluid dynamics, and multi-scale deep learning. |
| **Data Accessibility & Legality** | 20% | 9.0 | 1.80 | Free, public, authorized operational APIs (Copernicus Marine, Coriolis GDAC) under open scientific licenses. |
| **Feasibility of Demonstration** | 15% | 8.5 | 1.28 | Complete 3D volumetric fields can be rendered interactively via WebGL/deck.gl with slice-based inspection. |
| **Defensibility in Judging** | 20% | 9.5 | 1.90 | Quantitative depth-by-depth RMSE against independent ARGO floats eliminates subjective judging biases. |
| **National Strategic Impact** | 15% | 9.0 | 1.35 | Supports INCOIS, Indian Navy ASW, MoES Cyclone Early Warning, and fisheries management. |
| **Engineering De-risking Already Done** | 10% | 9.5 | 0.95 | End-to-end pilot pipeline, real 14-ch tensor, regridding, vertical interpolation, and training step verified. |
| **TOTAL WEIGHTED SCORE** | **100%** | — | **9.18 / 10.0** | **HIGHEST ACROSS ALL EVALUATED PS CANDIDATES** |

### 2. Comparison with Alternative PS Candidates
- *Candidate B (Healthcare Symptom Checker):* Disqualified due to lack of verifiable clinical ground truth, massive medical liability, and extreme competitive saturation (over 60 teams submitting standard LLM prompts).
- *Candidate C (Smart Traffic Signal Optimization):* Disqualified due to synthetic traffic simulation data (SUMO), lack of live municipal sensor feeds, and low technical defensibility.
- *SIH26066 Outcome:* Unambiguous winner based on verified data access, strategic relevance, and deep engineering substance.

"""

    p7 = """# PART 7: HOW DIFFICULT IS THIS? (BRUTALLY HONEST DIFFICULTY PROFILE)

We do not underestimate this problem statement. Reconstructing the interior of a turbulent, stratified fluid from surface measurements is mathematically ill-posed.

```
+---------------------------------------------------------------------------------------------------+
|                                  SIH26066 DIFFICULTY BREAKDOWN                                   |
+--------------------------+------------+-----------------------------------------------------------+
| Domain Component         | Difficulty | Primary Engineering & Scientific Bottlenecks             |
+--------------------------+------------+-----------------------------------------------------------+
| Data Engineering         | HIGH       | Multi-sensor spatial/temporal harmonization, regridding   |
| Oceanographic Physics    | HIGH       | Non-monotonic stratification, barrier layers, inversions  |
| Deep Learning Design     | MED-HIGH   | Multi-scale masked spatial encoding, depth conditioning   |
| Scientific Validation    | HIGH       | 4D spatial-temporal ARGO colocation, strict depth metrics |
| UI & System Integration  | MEDIUM     | Fast 3D volume slicing, real-time inference serving       |
+--------------------------+------------+-----------------------------------------------------------+
```

### 1. Specific Technical Complexities
1. **The Inversion Problem (Ill-Posed Mapping):** The mapping from 2D surface fields to 3D subsurface profiles is a Fredholm integral equation of the first kind—multiple subsurface thermal configurations can produce nearly identical surface signatures. The model must learn spatial context (gradients, vorticity, divergence) across hundreds of kilometers to infer vertical displacement.
2. **Coordinate & Grid Harmonization:** Ingesting 7 disparate products with native resolutions ranging from $0.05^\\circ$ (OSTIA SST) to $0.25^\\circ$ (Currents) requires robust conservative/bilinear regridding to the target $101 \\times 241$ grid without introducing interpolation artifacts.
3. **Missing Data & Swath Gaps:** Scatterometer and altimeter products contain orbital gaps. The model cannot simply fill missing values with zero without introducing artificial thermal shocks. Explicit validity masking is required.
4. **Computational Scale:** A 15-year daily corpus over $101 \\times 241$ with 14 input channels and 15 target depth channels represents $\\sim 5,475$ daily snapshots ($\\sim 15\\text{ GB}$ of packed float32 tensors).

"""

    p8 = """# PART 8: WHAT WE HAVE ALREADY PROVEN (AUDITED EVIDENCE)

Every milestone listed in this table has been executed, timed, and logged on real data in our workspace.

| Component / Claim | Mechanism & Test Script | Audited Findings & Measured Values | Audit Status |
| :--- | :--- | :--- | :---: |
| **SIH Requirements** | Official Problem Statement | Domain: $5^\\circ\\text{N}-30^\\circ\\text{N}, 45^\\circ\\text{E}-105^\\circ\\text{E}$; $0.25^\\circ$ grid ($101 \\times 241$); 15 depths. | `VERIFIED` |
| **3D GLORYS Pilot** | `scripts/download_glorys_3d_pilot.py` | Downloaded 44.74 MB NetCDF (`data/pilot/glorys_3d_pilot.nc`). 36 native depth levels ($0.494\\text{m}$ to $1062.44\\text{m}$). | `PILOT-VERIFIED` |
| **15-Depth Target Pipeline** | `scripts/process_real_3d_glorys.py` | Vertically interpolated to 15 standard depths and regridded to $101 \\times 241$ in $1.92\\text{s}$. Ocean cells: $11,854$ ($48.70\\%$). | `VERIFIED` |
| **0.25° Target Grid** | `scripts/verify_grid_regridding.py` | GLORYS regridded in $0.745\\text{s}$, OSTIA regridded in $0.739\\text{s}$. Verified dimensions: $101 \\times 241$. | `VERIFIED` |
| **SST Dataset** | `scripts/verify_sst_regridding.py` | Ingested real 414 MB OSTIA file (`METOFFICE-GLO-SST-L4-REP-OBS-SST`). Regridded to $101 \\times 241$. | `PILOT-VERIFIED` |
| **Genuine Satellite SSS** | `scripts/rebuild_real_xy_with_genuine_sss.py` | Downloaded 488 KB pilot (`cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c`). Weekly-to-daily interpolated into Channel 1 ($27.86$ to $44.13\\text{ psu}$). | `PILOT-VERIFIED` |
| **SSH Dataset** | `scripts/download_surface_pilots.py` | Downloaded 796 KB DUACS SLA NetCDF (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`). | `PILOT-VERIFIED` |
| **Surface Currents (U, V)** | `scripts/download_surface_pilots.py` | Downloaded 411 KB Multi-Obs surface currents (`cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m`). | `PILOT-VERIFIED` |
| **Surface Winds (U, V)** | `scripts/download_winds_pilot.py` | Downloaded 9.24 MB hourly scatterometer winds (`cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H`). Daily averaged. | `PILOT-VERIFIED` |
| **Real 14-Ch X & 15-Ch Y** | `scripts/rebuild_real_xy_with_genuine_sss.py` | Assembled 100% genuine observations into `data/pilot/sample_X_Y_real.pt` ($2.82\\text{ MB}$). Verified shapes $[1, 14, 101, 241]$ and $[1, 15, 101, 241]$. | `VERIFIED` |
| **Land Mask Loss Isolation** | `scripts/rebuild_real_xy_with_genuine_sss.py` | Injected $999,999.0$ into $12,487$ land pixels. Proved land gradient contribution is exactly $0.000000$. | `VERIFIED` |
| **Training-Step Sanity** | `scripts/rebuild_real_xy_with_genuine_sss.py` | Forward-backward pass executed across Pointwise MLP, Simple CNN, and OceanEmbedNet. Loss: $509.06$. | `VERIFIED` |
| **Independent ARGO Colocation**| `scripts/audit_argo_real.py` | Downloaded 6.14 MB ARGO NetCDF (`20221101_prof.nc`) from Coriolis GDAC. Filtered Float #34, colocated, and interpolated to 15 depths. | `PILOT-VERIFIED` |
| **Measured Compute Benchmark** | `scripts/benchmark_compute.py` | Measured on local CPU machine ($B=4$): Step time $5,133.83\\text{ ms}$, Peak RAM $1.66\\text{ GB}$. | `MEASURED` |
| **Adversarial Stress Tests** | `scripts/stress_test_break_everything.py` | 7 stress tests executed. Fixed IEEE 754 NaN loss bug and batch scaling bug. All 7 tests passing. | `VERIFIED` |

"""

    p9 = """# PART 9: WHAT WE HAVE NOT PROVEN YET (SCIENTIFIC BOUNDARIES)

In adherence to strict scientific honesty, the team explicitly defines what has **NOT** been proven yet. No team member or presentation slide may claim these as completed facts:

```
+---------------------------------------------------------------------------------------------------+
|                             CRITICAL SCIENTIFIC BOUNDARIES & LIMITS                               |
+---------------------------------------------------------------------------------------------------+
| 1. FINAL MODEL CONVERGENCE IS NOT PROVEN:                                                         |
|    We executed a single training step on a pilot tensor to verify pipeline data flow. This proves |
|    code correctness, NOT that the model has learned ocean physics or reached optimal weights.     |
|                                                                                                   |
| 2. BEATING BASELINES IS NOT YET PROVEN:                                                           |
|    OceanEmbed has not yet been benchmarked against Monthly Climatology, Linear Ridge, or U-Net   |
|    on a multi-year held-out test set. Superiority is a hypothesis to be experimentally validated. |
|                                                                                                   |
| 3. ACCURATE 1000m RECONSTRUCTION IS NOT PROVEN:                                                   |
|    Outputting a 15-depth tensor with 1000m dimensions is technically functional. However,        |
|    physical accuracy at 1000m remains an open scientific question due to surface decorrelation.   |
|                                                                                                   |
| 4. 15.6 YEARS OF HISTORICAL TENSORS DO NOT YET EXIST ON DISK:                                     |
|    The 2010–2025 overlap is verified in the Copernicus catalog. Downloading and pre-processing    |
|    the full multi-year training corpus is an upcoming Phase 1 execution task.                     |
+---------------------------------------------------------------------------------------------------+
```

Every claim we present to judges will strictly maintain the boundary between **verified engineering infrastructure** and **in-progress scientific training**.

"""

    p10 = """# PART 10: OUR SOLUTION — THE OCEANEMBED ARCHITECTURE

OceanEmbed is an end-to-end deep learning framework designed specifically to respect oceanographic physics and multi-modal satellite data properties.

### 1. Conceptual Architecture Diagram

```
+---------------------------------------------------------------------------------------------------+
|                                  OCEANEMBED SYSTEM ARCHITECTURE                                   |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  7 PHYSICAL SATELLITE VARIABLES (0.25° Grid)     7 BINARY VALIDITY MASKS (0=Invalid/Land, 1=Valid) |
|  [SST, SSS, SSH, Cur_U, Cur_V, Wind_U, Wind_V]   [Mask_SST, Mask_SSS, ..., Mask_WindV]             |
|                        \\                                  /                                       |
|                         \\                                /                                        |
|                          +------------------------------+                                         |
|                          |  14-CHANNEL CONCATENATED X   |                                         |
|                          |   Shape: [B, 14, 101, 241]   |                                         |
|                          +------------------------------+                                         |
|                                         |                                                         |
|                                         v                                                         |
|                          +------------------------------+                                         |
|                          | MASKED MULTI-SCALE U-NET     |                                         |
|                          | ENCODER (Partial/Masked Conv)|                                         |
|                          | Multi-scale receptive fields |                                         |
|                          +------------------------------+                                         |
|                                         |                                                         |
|                                         v                                                         |
|                          +------------------------------+                                         |
|                          | LATENT OCEAN EMBEDDING (Z)   |                                         |
|                          | Dimension: [B, 128, 101, 241]|                                         |
|                          | Compact dynamical footprint  |                                         |
|                          +------------------------------+                                         |
|                                         |                                                         |
|                                         v                                                         |
|                          +------------------------------+ <--- DEPTH CONDITIONING VECTOR          |
|                          | DEPTH-CONDITIONED PROFILE    |      Normalized Depth Coordinates z     |
|                          | DECODER (Cross-Attn / FiLM)  |      or Depth Embedding Embed(z)        |
|                          +------------------------------+                                         |
|                                         |                                                         |
|                                         v                                                         |
|                          +------------------------------+                                         |
|                          | 15-DEPTH SUBSURFACE TENSOR Y |                                         |
|                          |   Shape: [B, 15, 101, 241]   |                                         |
|                          | Depths: 0.49m down to 1000m  |                                         |
|                          +------------------------------+                                         |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

### 2. Core Functional Modules
1. **Masked Multi-Scale Encoder:** Ingests the 14-channel input. Uses gated or partial convolutions where invalid/land pixels do not bleed spurious edge signals into ocean cells. Extracts features across multiple spatial scales ($3 \\times 3$, $5 \\times 5$, $7 \\times 7$ receptive fields) to capture local frontal gradients and meso-scale eddies ($50-200\\text{ km}$).
2. **Latent Ocean Embedding ($Z$):** Compresses the 14 surface channels into a structured 128-dimensional latent representation per ocean grid cell. This embedding encapsulates the local thermodynamic and kinematic state (divergence, vorticity, steric height anomaly).
3. **Depth-Conditioned Decoder:** Rather than treating 15 depth levels as 15 independent, disconnected outputs, the decoder conditions on continuous or learned depth queries. It enforces vertical continuity, modeling the transition from the mixed layer through the thermocline into the abyssal ocean.

"""
    return [p6, p7, p8, p9, p10]
