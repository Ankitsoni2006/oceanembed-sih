# OceanEmbedNet Empirical Diagnostic Report: Subsurface Ocean Temperature Reconstruction (January 2020)

**Date**: 2026-09-12  
**Dataset**: Copernicus Marine In-Situ & Reanalysis (GLORYS12V1) + Satellite Observables (Jan 1–31, 2020)  
**Evaluation Protocol**: Out-of-sample chronological split (Train: Jan 1–24, 2020; Validation: Jan 25–31, 2020)  
**Target Domain**: North Indian Ocean (5°N–30°N, 45°E–105°E, 0.25° grid, 15 depths)  
**Data Artifacts**: `data/processed/chunk_real_2020_01_31day.pt`, `reports/real/oceanembed_diagnostics.json`

---

## Executive Summary

On the 7-day out-of-sample January 2020 validation set, `OceanEmbedNet` (1.34M parameters) achieved an overall RMSE of **1.6737°C** (Pearson $r = 0.9742$), performing substantially worse than the baseline `SimpleCNN` (35.6K parameters, RMSE **0.6284°C**, $r = 0.9961$) and pointwise `MLP` (19.4K parameters, RMSE **0.7201°C**, $r = 0.9949$).

This empirical diagnostic reveals that the underperformance is **not** random or caused by data corruption. Instead, it stems from three interacting structural and optimization failure modes:
1. **Severe Negative Bias from Hardcoded Climatology Prior Discrepancy**: Static initial priors in the thermocline (75m–200m) are **2.4°C to 3.2°C colder** than the true regional training mean, driving systematic negative prediction biases up to **-2.72°C**.
2. **Depth Conditioning Attenuation & Skip Connection Leakage**: Direct skip connections from the shallow encoder bypass the bottleneck and flood the shared decoder with surface satellite features (~27°C) across all 15 depth passes. The depth embedding gradient norm (**0.0095**) is an order of magnitude smaller than encoder gradients (**0.1190**), leaving depth conditioning too weak to override surface skips.
3. **Severe Overfitting / Capacity Mismatch on Small Temporal Sample**: On 24 training days, OceanEmbedNet's 1.34M parameters drove training loss down to **0.1463** while validation loss exploded to **2.8011**, whereas SimpleCNN maintained stable generalization (Train loss **0.3582**, Val loss **0.3948**).

---

## 1. Depth-Stratified Error & Distribution Analysis

Evaluating predictions across all 15 discrete vertical levels demonstrates how error varies by physical oceanographic regime:

| Depth (m) | Target Mean (°C) | Target Std (°C) | SimpleCNN Pred Mean (°C) | SimpleCNN Bias (°C) | SimpleCNN RMSE (°C) | OceanEmbed Pred Mean (°C) | OceanEmbed Bias (°C) | OceanEmbed RMSE (°C) | Climatology Prior (°C) | Prior Discrepancy (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0m** | 26.82 | 2.01 | 26.82 | +0.003 | **0.4309** | 27.09 | +0.273 | 0.7827 | 28.15 | +1.14 |
| **5m** | 26.78 | 1.99 | 26.79 | +0.011 | **0.4480** | 27.08 | +0.301 | 0.7399 | 28.06 | +1.08 |
| **10m** | 26.87 | 1.80 | 26.89 | +0.023 | **0.3897** | 27.44 | +0.570 | 0.7498 | 27.87 | +0.82 |
| **20m** | 26.94 | 1.70 | 26.97 | +0.026 | **0.4701** | 27.84 | +0.898 | 1.0233 | 27.47 | +0.34 |
| **30m** | 27.01 | 1.65 | 27.05 | +0.042 | **0.5126** | 28.12 | +1.110 | 1.2573 | 26.79 | -0.38 |
| **50m** | 26.82 | 1.69 | 26.84 | +0.017 | **0.6234** | 28.03 | +1.201 | 1.4473 | 25.04 | -1.88 |
| **75m** | 25.64 | 2.06 | 25.66 | +0.017 | **0.8507** | 24.68 | -0.954 | 1.5726 | 22.55 | -2.89 |
| **100m** | 23.35 | 2.35 | 23.39 | +0.037 | **0.9596** | 20.79 | **-2.559** | **2.8928** | 19.87 | **-3.15** |
| **125m** | 20.77 | 2.26 | 20.83 | +0.060 | **0.9751** | 18.04 | **-2.721** | **3.0275** | 17.27 | **-3.16** |
| **150m** | 18.44 | 2.00 | 18.55 | +0.110 | **0.8882** | 16.05 | **-2.390** | **2.6245** | 15.16 | **-3.01** |
| **200m** | 15.53 | 1.72 | 15.74 | +0.215 | **0.6738** | 13.36 | **-2.174** | **2.5044** | 13.00 | **-2.40** |
| **300m** | 13.07 | 1.57 | 13.28 | +0.211 | **0.5676** | 12.16 | -0.909 | 1.2434 | 10.53 | -2.50 |
| **500m** | 11.27 | 1.28 | 11.36 | +0.089 | **0.4240** | 11.27 | **+0.003** | **0.4204** | 8.45 | -2.78 |
| **700m** | 9.81 | 1.25 | 9.85 | +0.037 | **0.3990** | 10.42 | +0.614 | 0.7138 | 7.05 | -2.73 |
| **1000m** | 7.74 | 1.02 | 7.74 | -0.001 | **0.3716** | 9.01 | +1.273 | 1.3211 | 5.21 | -2.48 |

### Key Diagnostic Observations:
1. **Thermocline Failure Zone (100m–200m)**: OceanEmbedNet's RMSE peaks at **3.0275°C** at 125m, driven almost entirely by a **-2.721°C negative bias**. In contrast, SimpleCNN's bias at 125m is just **+0.060°C** with an RMSE of **0.9751°C**.
2. **Abyssal Crossover (500m)**: At 500m, OceanEmbedNet achieves **0.4204°C RMSE** and **+0.003°C bias**, outperforming SimpleCNN (0.4240°C). This occurs because the positive bias from decoder surface leakage precisely cancels the negative prior bias at this intermediate depth.
3. **Deep Positive Rebound (700m–1000m)**: Below 500m, the negative prior bias continues (-2.48°C at 1000m), but OceanEmbedNet flips to a **+1.273°C positive bias**, demonstrating that surface skip connection leakage is pulling deep ocean predictions upward toward surface temperatures (~27°C).

---

## 2. Climatology Prior Mismatch Analysis

In `src/models/oceanembed.py`, `self.climatology_prior` was initialized with static hardcoded constants:
```python
# Hardcoded prior initialization in OceanEmbedNet
self.climatology_prior = nn.Parameter(
    torch.tensor([28.15, 28.06, 27.87, 27.47, 26.79, 25.04, 22.55, 19.87,
                  17.27, 15.16, 13.00, 10.53, 8.45, 7.05, 5.21], dtype=torch.float32),
    requires_grad=True
)
```

Comparing these initial prior values against the true regional training mean (computed from 287,088 grid points over Jan 1–24, 2020) reveals large systematic discrepancies:
- **100m**: Prior = 19.87°C vs True Mean = 23.02°C $\rightarrow$ **-3.15°C discrepancy**
- **125m**: Prior = 17.27°C vs True Mean = 20.43°C $\rightarrow$ **-3.16°C discrepancy**
- **150m**: Prior = 15.16°C vs True Mean = 18.16°C $\rightarrow$ **-3.01°C discrepancy**
- **200m**: Prior = 13.00°C vs True Mean = 15.40°C $\rightarrow$ **-2.40°C discrepancy**

Because the model predicts an additive residual $\hat{Y}(d) = \text{Prior}(d) + \Delta T(X, d)$, and the shared decoder struggles to learn depth-specific offsets, the model inherits this ~3°C cold bias directly in the thermocline.

---

## 3. Gradient Flow & Component Dynamics

Measuring parameter gradient norms across architectural modules during training on the January dataset:

| Component | Parameter Count | Mean Gradient Norm | Relative Magnitude to Depth Embedding |
| :--- | :---: | :---: | :---: |
| **encoder_enc1** | 19,264 | **0.1190** | **12.6x** |
| **encoder_enc2** | 73,856 | **0.0891** | **9.4x** |
| **encoder_enc3** | 295,168 | 0.0545 | 5.8x |
| **decoder_dec3** | 295,040 | 0.0846 | 8.9x |
| **decoder_up3** | 65,600 | 0.0628 | 6.6x |
| **decoder_dec1** | 18,464 | 0.0527 | 5.6x |
| **bottleneck** | 590,080 | 0.0220 | 2.3x |
| **climatology_prior** | 15 | 0.0355 | 3.7x |
| **depth_embedding** | 1,920 | **0.0095** | **1.0x (Weakest)** |

### Architectural Implication:
- The depth embedding receives a gradient norm of only **0.0095**, by far the weakest signal in the entire network.
- Conversely, the shallow encoder layers (`enc1` = 0.1190, `enc2` = 0.0891) experience strong gradients driven by high-resolution surface spatial features.
- In the shared decoder loop, the skip connections carry high-gradient surface patterns directly into `dec1`, `dec2`, and `dec3`, completely overwhelming the weakly-conditioned depth vector injected at the coarse bottleneck.

---

## 4. Architectural Comparison: SimpleCNN vs OceanEmbedNet

| Design Dimension | SimpleCNN (Baseline) | OceanEmbedNet (Proposed) | Failure Analysis |
| :--- | :--- | :--- | :--- |
| **Parameter Count** | 35,663 parameters | 1,342,752 parameters | OceanEmbedNet has 37.6x more parameters, leading to severe overfitting on 24 daily samples. |
| **Output Mechanism** | Dedicated 15-channel output head: `Conv2d(64, 15, kernel_size=1)` | Shared single-channel decoder evaluated sequentially in a 15-iteration loop | SimpleCNN has dedicated weights and biases for each depth. OceanEmbedNet forces a single set of decoder weights to model all 15 depths simultaneously. |
| **Depth Conditioning** | Implicit: channel index $c \in [0..14]$ has dedicated convolution filters | Explicit: 128-d depth embedding concatenated at $12 \times 30$ bottleneck | Spatial resolution at bottleneck is compressed by 67.6x ($101 \times 241 \to 12 \times 30$). Fine-grained vertical stratification is lost. |
| **Skip Connections** | None (pure feedforward 4-layer conv) | 3 high-resolution U-Net skips (`e1`, `e2`, `e3`) from surface encoder | High-resolution skips inject surface temperature gradients into deep layer predictions without any depth-gating mechanism. |
| **Prior Formulation** | None: learns absolute scaled temperatures directly | Additive residual on hardcoded 1D tensor | Fixed priors were offset by -3.16°C in the thermocline; residual branch could not overcome this offset. |
| **Training Loss (Jan)** | 0.3582 | **0.1463** | OceanEmbedNet memorizes the training profiles (loss 0.1463). |
| **Validation Loss (Jan)**| **0.3948** (RMSE 0.6284°C) | **2.8011** (RMSE 1.6737°C) | Huge generalization gap (0.1463 vs 2.8011) indicates classic overparameterized overfitting on short temporal sequences. |

---

## 5. Summary of Root Causes

1. **Static Prior Initialization Error**: The hardcoded prior values in `src/models/oceanembed.py` had a -3.16°C cold bias relative to the North Indian Ocean winter climatology at 100m–150m.
2. **Ungated High-Resolution Skips**: The U-Net skip connections feed surface satellite features directly to the decoder output, causing deep predictions (700m–1000m) to be dragged warmer (+1.27°C bias at 1000m).
3. **Weak Bottleneck Depth Signal**: Concatenating a 128-d depth embedding onto a compressed $12 \times 30$ feature map is insufficient to guide a shared decoder across 1000m of vertical stratification when depth gradients are 12x weaker than surface skip gradients.
4. **Data-to-Parameter Imbalance**: Training a 1.34M parameter deep network on only 24 daily temporal snapshots (even with 11,962 spatial points per day) causes temporal memorization, as atmospheric and surface synoptic patterns on Jan 1–24 differ from Jan 25–31. SimpleCNN's compact 35.6K parameters acts as an effective inductive regularizer.
