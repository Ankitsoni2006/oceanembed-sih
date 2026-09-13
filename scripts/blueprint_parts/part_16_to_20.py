# -*- coding: utf-8 -*-
"""
SIH26066 Master Decision Guide - Parts 16 to 20
"""

def get_parts():
    p16 = """# PART 16: EXACT DATA SOURCES & ACCESS STATUS

Every dataset utilized in this project is an operational, peer-reviewed scientific product with an established data provider and unique digital identifier.

### 1. Master Dataset Register

| Variable / Stream | Product Identifier & Dataset ID | Data Provider | Native Resolution | Native Cadence | Role in Architecture | Verification Status |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **Sea Surface Temperature (SST)** | `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011`<br>`METOFFICE-GLO-SST-L4-REP-OBS-SST` | UK Met Office / Copernicus | $0.05^\\circ$ ($~5\\text{km}$) | Daily | Input Ch 0 (Value) & Ch 7 (Mask) | `PILOT-VERIFIED` |
| **Sea Surface Salinity (SSS)** | `MULTIOBS_GLO_PHY_SSS_L4_MY_015_015`<br>`cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c` | Copernicus Multi-Obs | $0.125^\\circ$ ($~12.5\\text{km}$) | 7-Day (Interpolated to Daily) | Input Ch 1 (Value) & Ch 8 (Mask) | `PILOT-VERIFIED` |
| **Sea Surface Height (SSH / SLA)**| `SEALEVEL_GLO_PHY_L4_MY_008_047`<br>`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` | DUACS / CLS / Copernicus | $0.125^\\circ$ ($~12.5\\text{km}$) | Daily | Input Ch 2 (Value) & Ch 9 (Mask) | `PILOT-VERIFIED` |
| **Surface Currents (uo, vo)** | `MULTIOBS_GLO_PHY_MYNRT_015_003`<br>`cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m` | Copernicus Multi-Obs | $0.25^\\circ$ ($~25\\text{km}$) | Daily | Input Ch 3, 4 (Values) & Ch 10, 11 (Masks) | `PILOT-VERIFIED` |
| **Surface Winds (u10, v10)** | `WIND_GLO_PHY_L4_MY_012_006`<br>`cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | KNMI / CERSAT / Copernicus | $0.125^\\circ$ ($~12.5\\text{km}$) | 1-Hour (Averaged to Daily) | Input Ch 5, 6 (Values) & Ch 12, 13 (Masks) | `PILOT-VERIFIED` |
| **Subsurface 3D Target Temperature**| `GLOBAL_MULTIYEAR_PHY_001_030`<br>`cmems_mod_glo_phy_my_0.083deg_P1D-m` (GLORYS12V1) | Mercator Ocean / Copernicus (DOI: 10.48670/moi-00021) | $0.083^\\circ$ ($~8\\text{km}$), 36 levels | Daily | Target Tensor Y (15 depths, 0.49m to 1000m) | `PILOT-VERIFIED` |
| **Independent In-Situ Ground Truth** | Coriolis Global Data Assembly Centre (GDAC)<br>`https://data-argo.ifremer.fr/` | Euro-Argo / Coriolis / WMO | Point CTD casts ($~1-2\\text{m}$ vertical) | ~10-day float cycle | Independent Test Benchmark (ARGO Floats) | `PILOT-VERIFIED` |

### 2. Transparent Clarification on INCOIS vs. Coriolis GDAC
- **The Empirical Reality:** During our Phase 10 validation audit, direct HTTP requests to the legacy INCOIS Live Access Server (LAS) endpoint (`https://las.incois.gov.in/`) returned **HTTP 404 (Not Found)** due to internal web server re-structuring at INCOIS.
- **The Verified Operational Alternative:** Coriolis GDAC is one of the two official WMO Global Data Assembly Centres for ARGO (along with US-GODAE). We connected directly to the Coriolis open HTTPS repository, downloaded multi-profile NetCDF `20221101_prof.nc` (6.14 MB), and successfully parsed real NIO float profiles.
- **Scientific Implication:** ARGO data is identical regardless of portal because all national DACs (including INCOIS) mirror their quality-controlled profiles to the Coriolis GDAC within 24–48 hours.

"""

    p17 = """# PART 17: DATA TEMPORAL OVERLAP (COMMON FEASIBLE WINDOW)

A machine learning model trained on multi-modal satellite inputs requires simultaneous temporal overlap across all 7 surface inputs and the 3D target.

### 1. Product Lifespans and Bounding Limiter
- **SST (OSTIA):** 1981-09-01 to 2026-present (>40 years).
- **SSH (DUACS Altimetry):** 1993-01-01 to 2026-present (>30 years).
- **Surface Currents (Multi-Obs):** 1993-01-01 to 2026-present (>30 years).
- **Surface Winds (Scatterometer):** 1999-08-01 to 2026-present (>25 years).
- **GLORYS12V1 Reanalysis:** 1993-01-01 to 2025-12-31 (>30 years).
- **SSS (Multi-Obs Satellite L4):** **2010-06-03 to 2025-12-25** (Limited by the launch and operational consolidation of SMOS and Aquarius/SMAP satellite salinity radiometers).

```
+---------------------------------------------------------------------------------------------------+
|                              COMMON TEMPORAL OVERLAP: 15.6 YEARS                                  |
|                                [2010-06-03  to  2025-12-25]                                       |
|                                     Total: 5,684 Days                                             |
+---------------------------------------------------------------------------------------------------+
```

### 2. Crucial Operational Distinction
- **What is Verified:** The existence, temporal continuity, and spatial coverage of all 8 products across this 15.6-year window is **OFFICIAL-CATALOG-VERIFIED** and **PILOT-VERIFIED** on sample dates.
- **What is NOT Claimed:** We do **NOT** claim that 5,684 days of processed tensors currently reside on our local hard drive. Downloading and processing this full corpus requires $\\sim 15\\text{ GB}$ of storage and is scheduled as Phase 1 of our execution roadmap.

"""

    p18 = """# PART 18: TRAINING STRATEGY & LOSS FORMULATION

### 1. Chronological Train / Validation / Test Splitting
Under no circumstances will the team use random K-fold cross-validation or random shuffling across time. Random splitting creates catastrophic temporal autocorrelation leakage (training on day $t$ and testing on day $t+1$).
We enforce strict chronological splitting:
- **Training Set (10 Years: 2011–2020):** $\\sim 3,650$ daily samples. Encompasses full decadal variability, positive/negative IOD events, and varying monsoon intensities.
- **Validation Set (2 Years: 2021–2022):** $\\sim 730$ daily samples. Used for early stopping, learning rate schedules, and hyperparameter tuning.
- **Held-Out Test Set (2 Years: 2023–2024):** $\\sim 730$ daily samples. Completely unseen during training. Evaluated depth-by-depth.
- **Near-Real-Time Blind Evaluation (2025):** 1 year reserved for blind evaluation against operational ARGO float trajectories.

### 2. Multi-Component Loss Function
The model is trained with a composite loss function that balances pointwise accuracy, outlier robustness, and physical vertical consistency:
$$\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{ocean}} + \\lambda_{\\text{grad}} \\mathcal{L}_{\\text{vert\\_grad}} + \\lambda_{\\text{smooth}} \\mathcal{L}_{\\text{laplace}}$$

Where:
1. **Masked Smooth L1 / Huber Loss ($\\mathcal{L}_{\\text{ocean}}$):**
   $$\\mathcal{L}_{\\text{ocean}} = \\frac{1}{N_{\\text{ocean}}} \\sum_{d=1}^{15} \\sum_{i, j \\in \\text{Ocean}} \\text{Huber}_\\delta\\left( Y_{d, i, j} - \\hat{Y}_{d, i, j} \\right)$$
   Huber loss (with $\\delta = 1.0^\\circ\\text{C}$) provides quadratic penalties for small errors while preventing extreme outliers (e.g., localized coastal thermal spikes) from dominating gradients.
2. **Vertical Gradient Regularization ($\\mathcal{L}_{\\text{vert\\_grad}}$):**
   $$\\mathcal{L}_{\\text{vert\\_grad}} = \\frac{1}{N_{\\text{ocean}}} \\sum_{d=1}^{14} \\sum_{i, j \\in \\text{Ocean}} \\left| \\frac{\\partial Y}{\\partial z} - \\frac{\\partial \\hat{Y}}{\\partial z} \\right|$$
   Enforces that the model learns the true physical thermocline gradient ($dT/dz$), preventing unphysical stepped or oscillatory temperature profiles.

### 3. Optimization Hyperparameters
- **Optimizer:** AdamW (weight decay $= 10^{-4}$, $\\beta_1 = 0.9, \\beta_2 = 0.999$).
- **Learning Rate Schedule:** Cosine Annealing with Warm Restarts ($\\\\eta_{\\text{max}} = 5 \\times 10^{-4}$, $\\\\eta_{\\text{min}} = 10^{-6}$, period $T_0 = 10$ epochs).
- **Precision:** Mixed Precision FP16 / BF16 via PyTorch `torch.cuda.amp.autocast()` to double throughput and halve VRAM requirements.

"""

    p19 = """# PART 19: BASELINES (COMPULSORY BENCHMARKING SUITE)

To prove to the SIH jury that OceanEmbed represents genuine machine learning innovation, we establish four mandatory benchmark baselines.

```
+---------------------------------------------------------------------------------------------------+
|                                MANDATORY BENCHMARKING HIERARCHY                                   |
+---------------------------------------------------------------------------------------------------+
| Baseline 1: Monthly Climatology (WOA / GLORYS Multi-Year Mean)  ---> Zero ML, Pure Physics Mean  |
| Baseline 2: Linear Ridge Regression (Pointwise Surface-to-Depth) ---> Simple Linear Mapping       |
| Baseline 3: Pointwise Multi-Layer Perceptron (MLP)             ---> Non-linear, No Spatial Contex|
| Baseline 4: Standard 2D U-Net Baseline                          ---> Spatial Context, Disjoint Ch|
| OceanEmbed (Proposed Model)                                     ---> Masked Multi-Scale + INR z   |
+---------------------------------------------------------------------------------------------------+
```

### 1. Specification of Baselines
1. **Monthly Climatology Baseline:** Computes the multi-year monthly mean 3D temperature $\\bar{T}_{m}(z, i, j)$ for each month $m \\in \\{1, \\dots, 12\\}$ from the historical training set. Any ML model that fails to outperform simple climatology is scientifically worthless.
2. **Linear Ridge Regression Baseline:** A separate linear ridge regressor trained per depth level $d$:
   $$\\hat{T}_d = \\sum_{k=0}^{6} w_{d, k} X_k + b_d$$
   Measures the linear component of surface-subsurface correlation.
3. **Pointwise MLP Baseline:** A 4-layer fully connected network (input 7 channels $\\to 64 \\to 128 \\to 64 \\to 15$ depth outputs) operating on isolated pixels without spatial convolution. Isolates the contribution of spatial receptive fields.
4. **Standard 2D U-Net Baseline:** A conventional 4-stage encoder-decoder U-Net outputting 15 static channels directly, without explicit validity masks or continuous depth conditioning.

### 2. Quantifiable Success Threshold
To declare OceanEmbed successful, it must achieve:
- Statistically significant RMSE reduction ($\\\\ge 20\\%$) over Monthly Climatology in the thermocline ($50-200\\text{m}$).
- Statistically significant RMSE reduction ($\\\\ge 15\\%$) over Pointwise MLP, proving the physical necessity of spatial context.

"""

    p20 = """# PART 20: EVALUATION FRAMEWORK & METRICS

Evaluation must be granular, stratified by depth, and mathematically rigorous. We do not report a single monolithic 'accuracy' percentage.

### 1. Primary Mathematical Metrics
Evaluated across all ocean grid cells for each depth level $d \\in \\{1, \\dots, 15\\}$:
1. **Root Mean Square Error (RMSE):**
   $$\\text{RMSE}_d = \\sqrt{\\frac{1}{N} \\sum_{i, j \\in \\text{Ocean}} \\left( Y_{d, i, j} - \\hat{Y}_{d, i, j} \\right)^2} \\quad [^\\circ\\text{C}]$$
2. **Mean Absolute Error (MAE):**
   $$\\text{MAE}_d = \\frac{1}{N} \\sum_{i, j \\in \\text{Ocean}} \\left| Y_{d, i, j} - \\hat{Y}_{d, i, j} \\right| \\quad [^\\circ\\text{C}]$$
3. **Mean Bias (Systematic Error):**
   $$\\text{Bias}_d = \\frac{1}{N} \\sum_{i, j \\in \\text{Ocean}} \\left( \\hat{Y}_{d, i, j} - Y_{d, i, j} \\right) \\quad [^\\circ\\text{C}]$$
4. **Pearson Correlation Coefficient ($r_d$):**
   $$r_d = \\frac{\\sum (Y_{d} - \\bar{Y}_d)(\\hat{Y}_d - \\bar{\\hat{Y}}_d)}{\\sqrt{\\sum (Y_d - \\bar{Y}_d)^2 \\sum (\\hat{Y}_d - \\bar{\\hat{Y}}_d)^2}}$$

### 2. Stratified Vertical Evaluation Regimes
The ocean exhibits drastically different thermal variability across the water column. We evaluate metrics across four distinct physical regimes:
- **Surface Mixed Layer ($0.49\\text{m} - 30\\text{m}$):** Dominated by air-sea fluxes. Expected RMSE: $<0.4^\\circ\\text{C}$.
- **Upper Thermocline ($50\\text{m} - 150\\text{m}$):** Highest physical variability ($dT/dz \\approx -0.15^\\circ\\text{C}/\\text{m}$). The primary benchmark zone for model differentiation. Expected RMSE: $<0.9^\\circ\\text{C}$.
- **Lower Thermocline & Intermediate Depth ($200\\text{m} - 500\\text{m}$):** Controlled by mesoscale eddy pumping and baroclinic Rossby waves. Expected RMSE: $<0.6^\\circ\\text{C}$.
- **Deep Abyssal Ocean ($750\\text{m} - 1000\\text{m}$):** Weak temporal variance, strong background stratification. Expected RMSE: $<0.3^\\circ\\text{C}$.

### 3. Extreme Event Evaluation (Cyclone Cold Wakes)
In addition to basin-wide statistics, we will evaluate model performance on case studies of named cyclones (e.g., Cyclone Tauktae in the Arabian Sea, Cyclone Amphan in the Bay of Bengal). We will plot cross-sectional transects across the storm track to verify if OceanEmbed accurately captures the **cyclone-induced thermocline shoaling and cold wake upwelling**.

"""
    return [p16, p17, p18, p19, p20]
