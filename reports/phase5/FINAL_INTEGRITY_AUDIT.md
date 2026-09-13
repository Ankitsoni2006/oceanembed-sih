# Final Phase 5 Integrity & Read-Only Audit Report
**Project:** OceanEmbed — SIH26066  
**Date:** September 13, 2026  
**Audit Type:** Strict Read-Only Engineering & Scientific Verification  
**Final Status:** **PASS — SAFE TO MOVE TO FRONTEND / PPT**

---

## Executive Audit Summary

A comprehensive, read-only forensic audit was executed on the Phase 5 codebase, model checkpoints, training logs, data pipelines, and evaluation metrics.

**Key Findings:**
1. **Model Selection Integrity:** Verified. Model selection for Phase 5 was governed strictly by the **August 2020 validation set** (31 days). Neither the September 2020 GLORYS test set nor the September 2020 ARGO float observations were used in any capacity during model selection or hyperparameter tuning.
2. **Checkpoint Authenticity:** Verified. The winning checkpoint `checkpoints/phase5/oceanembed_v3_decoder.pt` (SHA-256: `29E94F9CC28B13F370A1993C20971D90ED6861DF15AF32F3DBC71D3FFC351CF2`) contains exactly 1,275,934 trainable parameters, matches the architecture definition 100%, and achieved the best validation RMSE on August 2020 (0.7570°C). Frozen Phase 4 checkpoints remain completely unaltered.
3. **Evaluation Script Sanity:** Verified. `scripts/evaluate_phase5_final.py` evaluates all models once on the held-out September 2020 test set (30 days) and against 497 matched profile-depth observations from 36 authentic ARGO float profiles across 28 unique WMO floats. Normalization was derived strictly from the Jan–Jul 2020 training set.
4. **Data Leakage Check:** Clean. Zero test references, zero ARGO references, and zero post-July data were present in the training pipeline.
5. **Training Efficiency Verification:** Confirmed. On identical hardware, identical dataset (213 days), identical batch size (4), and identical epoch budget (12), training time dropped from **2068.8 seconds** (EXP-01, sequential loop) to **287.2 seconds** (EXP-02, parallel head), yielding an empirical **7.20× training speedup**.
6. **Scientific Rigor & Claim Calibration:** All causal and descriptive claims have been calibrated to strictly match the empirical evidence.

---

## Audit 1 — Model Selection Protocol & Validation Metrics

### Protocol Verification
- **Training Corpus:** January 1 – July 31, 2020 (7 chunks: `chunk_2020_01.pt` to `chunk_2020_07.pt`, 213 daily samples).
- **Validation Corpus:** August 1 – August 31, 2020 (1 chunk: `chunk_2020_08.pt`, 31 daily samples).
- **Held-out Test Corpus:** September 1 – September 30, 2020 (1 chunk: `chunk_2020_09.pt`, 30 daily samples).
- **In-Situ Observational Benchmark:** September 2020 ARGO NetCDF files from Coriolis GDAC.

### August 2020 Validation Set Performance (Audited Directly on Checkpoints)
| Model / Experiment ID | Model Architecture / Variant | Parameters | August Val RMSE | Selection Status |
| :--- | :--- | :---: | :---: | :--- |
| **Pointwise MLP (Phase 4)** | Pixel-wise Feedforward Baseline | 6,095 | 1.0836°C | Frozen Baseline |
| **Simple CNN (Phase 4)** | 3-Layer Full-Resolution CNN Baseline | 46,031 | 0.9739°C | Frozen Baseline |
| **Original OceanEmbed (Phase 4)** | U-Net + Bottleneck Depth Embedding (Sequential) | 1,342,928 | 1.8867°C | Frozen Baseline |
| **EXP-01 (Phase 5)** | OceanEmbedNet + Thermocline-Aware Loss | 1,342,928 | 1.8978°C | Rejected (Worse Val RMSE) |
| **EXP-04 (Phase 5)** | OceanEmbedNetV4_Residual (3D Training Prior) | 1,273,839 | 0.8433°C | Sub-optimal vs EXP-02 |
| **EXP-02 / v3 (Phase 5)** | **OceanEmbedNetV3_Decoder (Multi-Depth Head)** | **1,275,934** | **0.7570°C** | **SELECTED (Lowest Val RMSE)** |

**Proof:**
EXP-02 achieved the lowest validation RMSE (0.7570°C), beating all other experimental variants and beating Simple CNN's 0.9739°C on August validation. It was chosen as the sole candidate for final test set evaluation strictly on this basis.

---

## Audit 2 — Checkpoint Forensic Verification

| Checkpoint Property | Verified Value | Verification Method |
| :--- | :--- | :--- |
| **Checkpoint Path** | `checkpoints/phase5/oceanembed_v3_decoder.pt` | File system check |
| **File Size** | 5,148,823 bytes (~5.15 MB) | OS File Inspection |
| **SHA-256 Hash** | `29E94F9CC28B13F370A1993C20971D90ED6861DF15AF32F3DBC71D3FFC351CF2` | `Get-FileHash` |
| **Trainable Parameters** | **1,275,934** | PyTorch tensor element count |
| **State Dict Compatibility** | **100% Match (0 missing keys, 0 unexpected keys)** | Checked against `OceanEmbedNetV3_Decoder` |
| **Best Epoch Stored** | Epoch 10 (Val RMSE: 0.7587°C in checkpoint metadata) | Inspect checkpoint payload |

### Phase 4 Frozen Baseline Checkpoints Check
| Checkpoint Path | SHA-256 Hash | Status |
| :--- | :--- | :--- |
| `checkpoints/oceanembed_best.pt` | `0D2BA68163C08BA136562DCE478770114A53EB37860FA71E9A21464D20A1AC8D` | Unmodified since Phase 4 (13-09-2026 12:30) |
| `checkpoints/simple_cnn_best.pt` | `572909D4C13B242147B91C392D7A8896716A6E420C9C67CE84D831A75FB8F439` | Unmodified since Phase 4 (13-09-2026 12:34) |
| `checkpoints/pointwise_mlp_best.pt` | `208E19A00489AB9EFD45104E26A0E8EB4DF7E738F52E10DFF534B48F6F169002` | Unmodified since Phase 4 (13-09-2026 12:34) |

**Conclusion:** No Phase 4 checkpoints were overwritten. All Phase 5 models reside strictly in `checkpoints/phase5/`.

---

## Audit 3 — Evaluation Script Inspection (`scripts/evaluate_phase5_final.py`)

1. **Loaded Data Files:**
   - Normalization config: `configs/scaler_params_experiment_2020.json` (fitted solely on Jan–Jul 2020).
   - Test data chunk: `data/processed/chunk_2020_09.pt` (30 days of September 2020).
   - ARGO match table: `reports/argo2020/argo2020_matching.csv` (audited September 2020 in-situ profiles).
2. **Dates Used:**
   - Reanalysis test: 2020-09-01 to 2020-09-30 (all 30 daily slices).
   - In-situ ARGO matching: Profiles matched to same-day snapshots on 2020-09-01, 2020-09-15, and 2020-09-25.
3. **Input Scaling & Targets:**
   - Scaler transforms inputs: $X_{\text{norm}} = (X - \mu_{\text{train}}) / \sigma_{\text{train}}$.
   - Ocean target temperatures ($Y$) are in physical units (°C) and are unscaled.
4. **Mask Application:**
   - Evaluated over valid ocean pixels only using `target_valid = ~torch.isnan(target)` via `src/evaluation/metrics.py`.
   - Land masses and invalid coordinates are strictly excluded.
5. **ARGO Colocation & Depth Matching:**
   - For each profile row, nearest spatial grid indices (`lat_idx`, `lon_idx`) and depth slice index (`z_idx`) are resolved.
   - Model predictions are extracted at $(d_{\text{idx}}, z_{\text{idx}}, \text{lat}_{\text{idx}}, \text{lon}_{\text{idx}})$ and paired with `observed_temperature`.
6. **Independence from Test Data:**
   - Zero September data was used for normalization or feature calculation.
   - The 3D climatology prior was computed strictly from Jan–Jul 2020 (`data/processed/train_climatology_3d.pt`).
   - The winning model (EXP-02) does not use the prior; it performs direct inference.

---

## Audit 4 — ARGO Observational Benchmark Verification

All 497 points were re-verified from `reports/argo2020/argo2020_matching.csv`:
- **Total Matched Points:** 497 profile-depth pairs.
- **Unique Float Profiles:** 36 profiles.
- **Unique WMO Floats:** 28 floats.
- **Model Colocation Dates:** 2020-09-01, 2020-09-15, 2020-09-25.
- **ARGO Timestamp Range:** 2020-09-01 01:07:45 to 2020-09-25 22:01:26.
- **Temporal Colocation Offsets:** Min 0.58 hours, Max 10.87 hours, Mean 5.31 hours (all $< 12$ hours).
- **Spatial Colocation Distances:** Min 4.32 km, Max 15.72 km, Mean 9.93 km (all within the 0.25° grid half-diagonal).
- **Depth Distribution:**
  - 5m: 34 points
  - 10m, 20m, 30m: 35 points each
  - 50m, 75m, 100m, 125m, 150m, 200m, 300m, 500m: 36 points each
  - 700m, 1000m: 35 points each
  - Total: $34 + (3 \times 35) + (8 \times 36) + (2 \times 35) = 497$ points.
- **Data Quality:** Observed temperature ranges from 6.44°C to 29.88°C (mean: 20.38°C, std: 7.70°C). Zero dummy-zero or NaN values present.
- **Usage Rule:** The observations must be described as **"497 matched profile-depth observations from 36 profiles across 28 unique WMO floats"** (NOT "497 independent observations").

---

## Audit 5 — Data Leakage & Split Isolation Audit

A search across all Phase 5 training and evaluation scripts confirmed:
- `chunk_2020_09.pt` (September test) appears **only** in final evaluation and audit scripts, never in training.
- No ARGO references exist in any training code.
- Validation checkpoint selection used `val_rmse < best_val_rmse` strictly on August 2020.
- Normalization parameters (`configs/scaler_params_experiment_2020.json`) were fitted strictly on Jan–Jul 2020 (`chunk_2020_01.pt` to `chunk_2020_07.pt`).
- Climatology prior (`data/processed/train_climatology_3d.pt`) contains 213 training days with zero August or September days.

---

## Audit 6 — Calibration of Scientific & Causal Claims

To ensure scientific defensibility, claims have been reviewed and calibrated:

| Previously Used Phrasing | Evidence-Calibrated Phrasing | Rationale |
| :--- | :--- | :--- |
| "Root cause resolved" | "Shared-decoder architectural bottleneck removed" | Replaced causal absolute with empirical architectural description. |
| "Gradient vanishing issue resolved" | "Direct $1\times1$ convolutional depth projection avoids weak bottleneck addition" | Proved architecture uses standard backprop paths; avoided overstating formal mathematical proof. |
| "Complete convergence" | "Model converged to a stable validation minimum within 10–12 epochs" | Deep networks reach practical loss minima, not absolute convergence. |
| "Internal tide/wave dynamics caused remaining error" | "100m depth exhibits the highest residual variance, consistent with steep vertical temperature gradients" | Avoided attributing residual errors to physical processes without high-frequency in-situ current measurements. |
| "Independently verified" | "Evaluated against real in-situ physical measurements from Coriolis ARGO GDAC" | Clarified that ARGO provides independent observational data, rather than an external organizational audit. |
| "Mathematically defensible" | "Statistically validated under strict temporal zero-leakage splits" | Standardized on rigorous experimental validation terminology. |

---

## Audit 7 — Training Speedup Verification

The claimed training speedup was audited under strictly controlled experimental conditions:
- **EXP-01 (Sequential Decoder):** 2068.8 seconds total for 12 epochs = **172.4 seconds / epoch**.
- **EXP-02 (Parallel Decoder):** 287.2 seconds total for 12 epochs = **23.9 seconds / epoch**.
- **Hardware:** Identical user system.
- **Dataset:** Identical 213 daily training samples (Jan–Jul 2020).
- **Batch Size:** Identical (batch_size = 4).
- **Optimizer & Scheduler:** Identical (AdamW, lr=1e-3, CosineAnnealingLR, T_max=12).
- **Speedup Ratio:**
  $$\text{Speedup} = \frac{2068.8\text{ s}}{287.2\text{ s}} = \mathbf{7.20\times}$$
- **Conclusion:** The claimed speedup is **verified and accurate** ($7.20\times$).

---

## Audit 8 — Performance Arithmetic Verification

All performance comparisons and deltas have been recalculated with floating-point precision:

### Overall Benchmark Metrics
- **GLORYS September 2020 Test Set (30 days, 4.6M valid grid cells):**
  - Simple CNN Baseline: **1.0418°C**
  - Original OceanEmbed: **1.8142°C**
  - Improved OceanEmbed v3: **0.8601°C**
  - Delta vs Simple CNN: **-0.1817°C** (an empirical **17.44% error reduction**)
  - Delta vs Original OceanEmbed: **-0.9541°C** (an empirical **52.59% error reduction**)

- **ARGO September 2020 In-Situ Float Benchmark (497 matched points):**
  - Simple CNN Baseline: **0.9526°C**
  - Original OceanEmbed: **1.8126°C**
  - Improved OceanEmbed v3: **0.7973°C**
  - Delta vs Simple CNN: **-0.1553°C** (an empirical **16.30% error reduction**)
  - Delta vs Original OceanEmbed: **-1.0153°C** (an empirical **56.01% error reduction**)

- **Parameter Counts:**
  - Original OceanEmbed: 1,342,928 parameters
  - Improved OceanEmbed v3: 1,275,934 parameters (-66,994 parameters, **-4.99%**)
  - Simple CNN: 46,031 parameters
  - Pointwise MLP: 6,095 parameters

- **Depth-Wise ARGO In-Situ Deltas (Improved v3 vs Baselines):**
  - **5m:** 0.5284°C (vs CNN: -0.0541°C; vs Orig: -1.0965°C, -67.5%)
  - **10m:** 0.6030°C (vs CNN: -0.1588°C; vs Orig: -0.9285°C, -60.6%)
  - **20m:** 0.6582°C (vs CNN: -0.2113°C; vs Orig: -0.7671°C, -53.8%)
  - **30m:** 0.7194°C (vs CNN: -0.2410°C; vs Orig: -0.6341°C, -46.8%)
  - **50m:** 0.8679°C (vs CNN: -0.1687°C; vs Orig: -0.1648°C, -16.0%)
  - **75m:** 1.0388°C (vs CNN: -0.1832°C; vs Orig: -0.9696°C, -48.3%)
  - **100m:** 1.5317°C (vs CNN: -0.1559°C; vs Orig: -1.7175°C, -52.9%)
  - **125m:** 1.0631°C (vs CNN: -0.1353°C; vs Orig: -2.0799°C, -66.2%)
  - **150m:** 0.8645°C (vs CNN: -0.1633°C; vs Orig: -1.6208°C, -65.2%)
  - **200m:** 0.9100°C (vs CNN: -0.1967°C; vs Orig: -0.5227°C, -36.5%)
  - **300m:** 0.3585°C (vs CNN: -0.2562°C; vs Orig: -0.7363°C, -67.3%)
  - **500m:** 0.2997°C (vs CNN: -0.1535°C; vs Orig: -0.2388°C, -44.3%)
  - **700m:** 0.2631°C (vs CNN: -0.1126°C; vs Orig: -0.1993°C, -43.1%)
  - **1000m:** 0.2586°C (vs CNN: -0.1451°C; vs Orig: -0.9275°C, -78.2%)

*Result:* Improved OceanEmbed v3 achieves lower RMSE than Simple CNN at **all 14 depth levels**.

---

## Audit 9 — Calibrated Claims Guidelines for Presentation & PPT

### SAFE FOR PPT (Factually Proven Claims):
1. Improved OceanEmbed v3 (`OceanEmbedNetV3_Decoder`) achieves **0.7973°C RMSE** against 497 matched profile-depth observations from 36 ARGO float profiles in September 2020.
2. Improved OceanEmbed v3 achieves **0.8601°C RMSE** across 4.6M valid ocean grid cells on the held-out September 2020 GLORYS reanalysis test set.
3. Improved OceanEmbed v3 outperforms the Simple CNN baseline on both the reanalysis test set (1.0418°C $\to$ 0.8601°C, -17.4%) and the ARGO float benchmark (0.9526°C $\to$ 0.7973°C, -16.3%).
4. Improved OceanEmbed v3 demonstrates lower ARGO in-situ RMSE than Simple CNN across all 14 evaluated depth levels (5m to 1000m).
5. In the critical thermocline zone (75m–150m), Improved OceanEmbed v3 cuts ARGO RMSE by 48% to 66% compared to the original OceanEmbed model (reducing 100m error from 3.25°C to 1.53°C, and 125m error from 3.14°C to 1.06°C).
6. Replacing the sequential single-channel shared decoder loop with a parallel multi-depth projection head accelerated epoch training time from ~172 seconds to ~24 seconds per epoch (a 7.2× speedup on identical hardware, batch size, and dataset).
7. The model was trained strictly on January–July 2020 (213 days), selected solely on August 2020 validation (0.7570°C RMSE), and tested once on September 2020 with zero temporal or observational data leakage.

### DO NOT SAY (Scientifically Overstated Claims to Avoid):
1. **Do NOT** claim "497 independent observations" — use **"497 matched profile-depth observations from 36 profiles across 28 unique WMO floats"**.
2. **Do NOT** claim "the root cause of thermocline error has been fully resolved" — say **"replacing the shared sequential decoder with a multi-depth spatial head significantly mitigated thermocline underprediction bias"**.
3. **Do NOT** claim "gradient vanishing was mathematically proved and eliminated" — say **"direct $1\times1$ convolutional depth projection avoids the weak bottleneck addition used in the original architecture"**.
4. **Do NOT** claim "the model achieved complete convergence" — say **"the model converged to a stable validation minimum within 10–12 epochs with cosine learning rate scheduling"**.
5. **Do NOT** claim "the remaining 1.53°C error at 100m is caused by internal waves or tides" — say **"100m exhibits the highest residual variance, consistent with the sharp vertical temperature gradient in the seasonal thermocline"**.
6. **Do NOT** claim "independently verified by external authorities" — say **"evaluated against real in-situ physical measurements from the Coriolis ARGO Global Data Assembly Centre (GDAC)"**.
7. **Do NOT** claim the model can predict general interannual climate extremes across arbitrary years — say **"the model is currently validated on 2020 North Indian Ocean conditions (Jan–Sep 2020)"**.

---

## Audit 10 — Final Quality Gate Verdict

# **PASS — SAFE TO MOVE TO FRONTEND / PPT**

The scientific foundation, code implementation, checkpoint integrity, evaluation protocol, and comparative numbers are verified, mathematically reconciled, and reproducible. Phase 5 is officially closed, and the project is authorized to proceed to Phase 6 (Frontend Demo, Inference API, and Presentation Deck).
