# -*- coding: utf-8 -*-
"""
SIH26066 Master Decision Guide - Parts 11 to 15
"""

def get_parts():
    p11 = """# PART 11: WHY WE USE MASKS (LAND & MISSING OBSERVATION BOUNDARIES)

In terrestrial computer vision, every pixel in an image is usually valid. In satellite oceanography, this assumption fails catastrophically.

### 1. The Reality of Ocean Data Gaps
1. **The Land Mask Barrier:** In the North Indian Ocean domain ($5^\\circ\\text{N}-30^\\circ\\text{N}, 45^\\circ\\text{E}-105^\\circ\\text{E}$), the Indian subcontinent, Arabian Peninsula, Southeast Asia, and islands occupy **12,487 grid cells**, or **51.30% of the entire grid**. Only **11,854 grid cells (48.70%)** are actual ocean water.
   - If land pixels are filled with zeros or arbitrary values, standard convolutional filters will calculate massive spurious gradients at coastlines.
   - The model will spend its parameter budget memorizing coastal outlines rather than ocean physics.
2. **Cloud Contamination in Optical/Infrared SST:** Infrared satellite radiometers cannot penetrate clouds. During the summer monsoon (June–September), cloud cover in the Bay of Bengal can exceed 80%, creating missing observation swaths.
3. **Orbital Altimeter & Scatterometer Swath Gaps:** Radar altimeters only measure along narrow nadir ground tracks. Multi-satellite L4 products interpolate these gaps, but coastal and high-latitude coverage varies.

### 2. The Dual-Channel Solution: Value + Validity Mask
For each of the 7 physical surface variables, we provide two coupled channels:
$$\\text{Input Channel } i = \\text{Physical Value (normalized)}$$
$$\\text{Input Channel } i+7 = \\text{Binary Validity Mask } (1.0 = \\text{Valid Ocean Observation}, 0.0 = \\text{Missing or Land})$$

```
Channel 0: SST Value            ---> Channel 7:  SST Mask
Channel 1: SSS Value            ---> Channel 8:  SSS Mask
Channel 2: SSH Value            ---> Channel 9:  SSH Mask
Channel 3: Current U Value      ---> Channel 10: Current U Mask
Channel 4: Current V Value      ---> Channel 11: Current V Mask
Channel 5: Wind U Value         ---> Channel 12: Wind U Mask
Channel 6: Wind V Value         ---> Channel 13: Wind V Mask
```

### 3. Mathematical Formulation of Masked Loss
Let $Y_{d, i, j}$ be the ground truth temperature at depth $d$ and coordinate $(i, j)$, $\\hat{Y}_{d, i, j}$ be the model prediction, and $M_{i, j} \\in \\{0, 1\\}$ be the ocean validity mask ($M_{i, j} = 1$ if ocean, $0$ if land). The masked mean squared error loss is defined as:
$$\\mathcal{L}_{\\text{ocean}} = \\frac{\\sum_{d=1}^{15} \\sum_{i=1}^{H} \\sum_{j=1}^{W} M_{i, j} \\cdot \\left( Y_{d, i, j} - \\hat{Y}_{d, i, j} \\right)^2}{\\sum_{i=1}^{H} \\sum_{j=1}^{W} M_{i, j} \\cdot 15}$$

**Mathematical Proof of Land Isolation:**
$$\\frac{\\partial \\mathcal{L}_{\\text{ocean}}}{\\partial \\hat{Y}_{d, i_{\\text{land}}, j_{\\text{land}}}} = 0$$
Because $M_{i_{\\text{land}}, j_{\\text{land}}} = 0$, any prediction or corruption over land yields exactly zero loss and zero gradient update.

"""

    p12 = """# PART 12: WHY OCEAN EMBEDDING? (THE LATENT SPACE CONCEPT)

### 1. Demystifying the "Ocean Embedding"
We must be clear with our team and the SIH jury:
> **The Latent Ocean Embedding is NOT a metaphysical or "true hidden ocean state."**
> Rather, it is a **learned, compact, multi-scale feature representation** of the surface dynamical footprint that provides the optimal conditioning context for vertical profile reconstruction.

### 2. The Information Bottleneck
A direct pointwise mapping from 7 surface values $(T_{\\text{sfc}}, S_{\\text{sfc}}, \\eta, u, v, w_u, w_v)$ to 15 subsurface depths lacks spatial context. For instance, a sea surface height anomaly $\\eta = +15\\text{ cm}$ could indicate:
- A warm-core anti-cyclonic eddy with a deep thermocline bowl.
- A coastally trapped Kelvin wave propagating along the boundary.
- A broad seasonal steric warming event.

A pointwise model cannot distinguish between these three physical regimes because it sees only a single pixel.
**The Ocean Embedding Network ($E_\\theta$)** looks at an extensive spatial receptive field ($150-300\\text{ km}$ across the $101 \\times 241$ grid) using multi-scale convolutional kernels. It computes spatial derivatives:
- Relative vorticity: $\\zeta = \\frac{\\partial v}{\\partial x} - \\frac{\\partial u}{\\partial y}$
- Horizontal divergence: $\\delta = \\frac{\\partial u}{\\partial x} + \\frac{\\partial v}{\\partial y}$
- Thermal and haline frontal gradients: $|\\nabla \\text{SST}|, |\\nabla \\text{SSS}|$

The encoder compresses these multi-scale spatial gradients into a 128-dimensional embedding vector $Z_{i, j} \\in \\mathbb{R}^{128}$ at each grid cell. This embedding represents the local dynamical regime (e.g., eddy center, boundary current, upwelling zone) before vertical reconstruction begins.

"""

    p13 = """# PART 13: WHY DEPTH-CONDITIONED DECODING?

### 1. The Pitfall of Disjoint Multi-Head Regressors
A naive neural network approach to 3D ocean reconstruction uses a standard 2D CNN that directly outputs 15 output channels:
$$\\hat{Y} = \\text{CNN}(X) \\in \\mathbb{R}^{15 \\times 101 \\times 241}$$
Why is this suboptimal?
1. **Treats Depths as Disjoint Classes:** The final $1 \\times 1$ convolutional layer simply projects features into 15 static channels. The network has no intrinsic mathematical concept that channel 3 ($20\\text{m}$) is physically adjacent to channel 4 ($30\\text{m}$).
2. **Inability to Generalize to Arbitrary Depths:** If a user or naval operator requests temperature at $125\\text{m}$ (between standard levels $100\\text{m}$ and $150\\text{m}$), a static 15-channel CNN cannot evaluate it without external interpolation.
3. **Gradient Decoupling:** Errors in the thermocline do not backpropagate smoothly into the mixed layer representation.

### 2. The Depth-Conditioned Continuous Formulation
OceanEmbed formulates subsurface reconstruction as a **continuous depth-conditioned function**:
$$\\hat{T}(z; i, j) = \\mathcal{D}_\\phi\\left( Z_{i, j}, \\gamma(z) \\right)$$
Where:
- $Z_{i, j} \\in \\mathbb{R}^{128}$ is the latent ocean embedding at spatial coordinate $(i, j)$.
- $z \\in [0, 1000\\text{m}]$ is the target vertical depth.
- $\\gamma(z) \\in \\mathbb{R}^{D_z}$ is a positional depth encoding (e.g., Fourier sinusoidal embeddings or learned continuous MLP embeddings):
  $$\\gamma(z) = \\left[ \\sin\\left(\\frac{2^0 \\pi z}{z_{\\text{max}}}\\right), \\cos\\left(\\frac{2^0 \\pi z}{z_{\\text{max}}}\\right), \\dots, \\sin\\left(\\frac{2^{K-1} \\pi z}{z_{\\text{max}}}\\right), \\cos\\left(\\frac{2^{K-1} \\pi z}{z_{\\text{max}}}\\right) \\right]$$
- $\\mathcal{D}_\\phi$ is a profile decoder using FiLM (Feature-wise Linear Modulation) or cross-attention.

**Key Oceanographic Advantage:** Because the decoder is conditioned on $z$, it enforces vertical smoothness and continuity. It acts as an implicit neural representation (INR) along the vertical column, ensuring that reconstructed temperature profiles behave as smooth physical water columns rather than noisy, uncorrelated layers.

"""

    p14 = """# PART 14: WHY THIS IS DIFFERENT (LITERATURE POSITIONING & NOVELTY AUDIT)

A critical requirement for winning SIH is establishing **rigorous positioning against existing literature**, rather than making empty marketing claims like *"nobody has ever done this before."*

### 1. Comprehensive Literature Comparison

| Research Architecture / Paper | Core Methodology | Major Strengths | Inherent Limitations | OceanEmbed Differentiation |
| :--- | :--- | :--- | :--- | :--- |
| **Pointwise Random Forest / MLP** *(Ali et al., 2004; Su et al., 2015)* | Ingests single-pixel SST, SSS, SSH into tabular regression. | Fast, simple baseline; low compute footprint. | Zero spatial context; cannot detect eddies or front-driven upwelling; high thermocline error. | Multi-scale U-Net encoder extracts spatial gradients, divergence, and vorticity across $300\\text{km}$ neighborhoods. |
| **ConvLSTM Subsurface Models** *(Meng et al., 2021; Song et al., 2022)* | Recurrent convolutional cells modeling spatio-temporal sequences. | Captures temporal memory and seasonal cycles. | Extremely heavy memory footprint; slow training; suffers from error accumulation over long rollouts. | Replaces recurrence with compact latent embedding and temporal positional encoding; fast feedforward inference. |
| **Convformer / FWinFormer** *(Li et al., 2023; Wang et al., 2024)* | Hybrid convolution-windowed vision transformer. | Strong long-range spatial attention across ocean basins. | Quadratic attention complexity; struggles with irregular coastlines; ignores land masks. | Explicit dual-channel validity masks prevent land contamination; masked loss isolates ocean domain. |
| **Graph Neural Networks (GNN)** *(Sun et al., 2023)* | Models unstructured float networks as graph nodes. | Naturally handles irregularly spaced ARGO floats. | Extremely slow inference; cannot efficiently generate dense, regular $101 \\times 241$ 3D grid fields. | Formulates regular grid mapping from satellite L4 fields, reserving ARGO strictly for independent validation. |
| **Deep Evidential Regression** *(Amini et al., 2020; Charpentier, 2024)* | Estimates epistemic and aleatoric uncertainty via Normal-Inverse-Gamma priors. | Provides uncertainty bounds for predictions. | Difficult to calibrate in highly non-linear stratified layers; high training instability. | Planned as an uncertainty head on top of the frozen latent embedding in Phase 6. |
| **3D U-Net++ / TS-Cast** *(Chen et al., 2022; Zhou et al., 2025)* | 3D volumetric convolutions projecting surface down to 3D grid. | Directly models 3D continuity. | Massive compute requirements; treats vertical axis $z$ identically to horizontal axes $x, y$ despite vertical scale ($1\\text{km}$) being $100\\times$ smaller than horizontal ($1000\\text{km}$). | Depth-conditioned continuous decoder treats vertical stratification with physics-appropriate asymmetry. |

### 2. Our Specific Scientific Differentiation
Our differentiation is **not** that we invented deep learning for oceanography. Our differentiation is:
1. **Targeted NIO Domain Formulation:** Specifically tailored for the complex, monsoon-reversing North Indian Ocean ($5^\\circ\\text{N}-30^\\circ\\text{N}, 45^\\circ\\text{E}-105^\\circ\\text{E}$).
2. **7-Variable Multi-Modal Synergy:** Uniquely fusing SST, genuine satellite SSS, SLA, surface currents ($u, v$), and scatterometer winds ($u, v$).
3. **Rigorous Dual-Channel Masking:** Mathematically isolating the 51.30% land area to prevent gradient contamination.
4. **Independent ARGO Ground Truth Isolation:** Maintaining strict separation between training reanalysis (GLORYS) and independent physical CTD validation (Coriolis GDAC).

"""

    p15 = """# PART 15: COMPLETE DATA PIPELINE (END-TO-END SPECIFICATION)

The data pipeline transforms raw, multi-source NetCDF-4 and GRIB files into standardized, machine-learning-ready PyTorch tensors.

```
+---------------------------------------------------------------------------------------------------+
|                                  END-TO-END DATA PROCESSING FLOW                                  |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [COPERNICUS MARINE DATA STORE]                                    [CORIOLIS GDAC ARGO SERVER]   |
|  - SST (OSTIA L4, 0.05°)                                           - Global Profiling Floats     |
|  - SSS (Multi-Obs OI L4, 7-Day, 0.125°)                            - Indian Ocean Deployments    |
|  - SSH (DUACS SLA L4, 0.125°)                                      - Real CTD Temperature/Depth  |
|  - Currents (Multi-Obs L4, 0.25°)                                                                |
|  - Winds (Scatterometer L4, 0.125°, 1H)                                                           |
|  - GLORYS12V1 (3D Reanalysis, 0.083°, 36 levels)                                                 |
|                        |                                                         |                |
|                        v                                                         v                |
|  +--------------------------------------------+            +------------------------------------+ |
|  | STEP 1: SPATIAL SUBSETTING & TIME FILTER   |            | STEP 1B: ARGO FLOAT FILTERING      | |
|  | Crop bounding box: 5°N-30°N, 45°E-105°E    |            | Filter NIO domain (5-30N, 45-105E) | |
|  | Align timestamps to daily Julian Day       |            | Extract Quality Flag = 1 or 2      | |
|  +--------------------------------------------+            +------------------------------------+ |
|                        |                                                         |                |
|                        v                                                         v                |
|  +--------------------------------------------+            +------------------------------------+ |
|  | STEP 2: REGRIDDING & VERTICAL INTERPOLATION|            | STEP 2B: ARGO VERTICAL INTERP      | |
|  | - Horizontal: Bilinear to 0.25° (101x241)  |            | Interpolate raw CTD pressure/temp  | |
|  | - SSS: Weekly-to-daily linear interpolation|            | to 15 standard depths (0.49-1000m) | |
|  | - Winds: Hourly vectors averaged to daily  |            +------------------------------------+ |
|  | - GLORYS: 36 levels -> 15 standard depths  |                                  |                |
|  +--------------------------------------------+                                  |                |
|                        |                                                         |                |
|                        v                                                         |                |
|  +--------------------------------------------+                                  |                |
|  | STEP 3: MASK GENERATION & NORMALIZATION    |                                  |                |
|  | - Compute 7 binary validity masks (0/1)    |                                  |                |
|  | - Robust Z-score normalization per channel |                                  |                |
|  | - Land cells forced to 0.0 with Mask=0.0   |                                  |                |
|  +--------------------------------------------+                                  |                |
|                        |                                                         |                |
|                        v                                                         |                |
|  +--------------------------------------------+                                  |                |
|  | STEP 4: TENSOR PACKING & SPLIT             |                                  |                |
|  | Input Tensor X:  [B, 14, 101, 241]         |                                  |                |
|  | Target Tensor Y: [B, 15, 101, 241]         |                                  |                |
|  | Split: Train (2011-20), Val (21-22), Test  |                                  |                |
|  +--------------------------------------------+                                  |                |
|                        |                                                         |                |
|                        v                                                         v                |
|  +----------------------------------------------------------------------------------------------+ |
|  | STEP 5: PYTORCH DATALOADER & EVALUATION HARNESS                                              | |
|  | - Batching with pinned memory                                                                | |
|  | - Masked Huber Loss + Vertical Gradient Regularization                                       | |
|  | - Dual Evaluation: Reanalysis Test Set (GLORYS) & In-Situ Independent Set (ARGO)             | |
|  +----------------------------------------------------------------------------------------------+ |
+---------------------------------------------------------------------------------------------------+
```

### 2. Processing Specifications & Timing
- **Regridding Engine:** `xarray` and `scipy.interpolate.RegularGridInterpolator`. Benchmark regrid time for a single global daily snapshot is **$0.745\\text{s}$** (GLORYS) and **$0.739\\text{s}$** (OSTIA).
- **Vertical Target Interpolation:** 1D piecewise cubic Hermite or monotonic spline interpolation across 36 native GLORYS depths ($0.494\\text{m}$ to $1062.44\\text{m}$) down to the 15 SIH target depths. Executed in **$1.92\\text{s}$** per time slice.

"""
    return [p11, p12, p13, p14, p15]
