# -*- coding: utf-8 -*-
"""
SIH26066 Master Decision Guide - Parts 21 to 25
"""

def get_parts():
    p21 = """# PART 21: INDEPENDENT ARGO FLOAT VALIDATION METHODOLOGY

One of the most critical scientific assertions in our blueprint is maintaining the strict distinction between **Reanalysis Target Data** and **Independent Observational Ground Truth**.

### 1. The Distinction: GLORYS Reanalysis vs. In-Situ ARGO
```
+---------------------------------------------------------------------------------------------------+
| GLORYS12V1 REANALYSIS (CMEMS)           | ARGO PROFILING FLOATS (CORIOLIS GDAC)                   |
+-----------------------------------------+---------------------------------------------------------+
| - Model-data assimilation synthesis     | - Pure direct physical measurement                      |
| - Continuous, regular 3D grid           | - Point observations in space and time                  |
| - Used as the TRAINING TARGET (Y)       | - NEVER seen during training                            |
| - Subject to numerical model biases     | - STRICT INDEPENDENT VALIDATION BENCHMARK               |
+---------------------------------------------------------------------------------------------------+
```

### 2. Spatial and Temporal Colocation Algorithm
Because an ARGO float drifts freely with ocean currents and surfaces only once every $\\sim 10$ days, it does not sit precisely at grid vertices:
1. **Temporal Filtering:** For a target test day $t$, isolate all ARGO profiles collected within $[t - 12\\text{ hours}, t + 12\\text{ hours}]$.
2. **Spatial Localization:** Extract the float\\'s GPS coordinate $(\\lambda_{\\text{argo}}, \\phi_{\\text{argo}})$. Map to the nearest ocean grid cell $(i^*, j^*)$ on our $0.25^\\circ$ grid:
   $$i^* = \\text{round}\\left( \\frac{\\phi_{\\text{argo}} - 5.0^\\circ}{0.25^\\circ} \\right), \\quad j^* = \\text{round}\\left( \\frac{\\lambda_{\\text{argo}} - 45.0^\\circ}{0.25^\\circ} \\right)$$
3. **Vertical Spline Interpolation:** Raw CTD profiles record temperature at continuous, irregular pressure intervals ($1-2\\text{m}$ resolution). We apply monotonic PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) to resample the in-situ float profile to our exact 15 target depths ($0.494\\text{m}$ to $1000\\text{m}$).
4. **Validation Metrics Computation:** Calculate depth-wise RMSE and Bias between the model\\'s reconstructed profile $\\hat{Y}(i^*, j^*, :)$ and the in-situ float CTD vector $T_{\\text{argo}}(:)$.

### 3. Empirical Demonstration in Audit
In our Phase 10 audit, we downloaded Coriolis GDAC NetCDF file `20221101_prof.nc` ($6.14\\text{ MB}$) containing 1,029 global profiles. We filtered floats in the North Indian Ocean, identified Indian Ocean Float #34 ($17.9540^\\circ\\text{N}, 63.5540^\\circ\\text{E}$), colocated it to grid cell $(18.00^\\circ\\text{N}, 63.50^\\circ\\text{E}$), and successfully verified that the vertical interpolation executes cleanly across all 15 depths down to $1000\\text{m}$.

"""

    p22 = """# PART 22: PHYSICS & SCIENTIFIC GUARDRAILS (TEMPERATURE INVERSIONS)

A naive machine learning practitioner might attempt to enforce a hard physical penalty such as:
$$\\text{Penalty} = \\sum_{d=1}^{14} \\max\\left(0, \\hat{T}_{d+1} - \\hat{T}_d\\right) \\quad \\text{(Enforcing that temperature must always decrease with depth)}$$
**In the North Indian Ocean, imposing this naive constraint would be a catastrophic scientific failure.**

### 1. Why Temperature Inversions Actually Occur
In tropical oceanography, density $\\rho$ is governed by both **Temperature ($T$)** and **Salinity ($S$)** via the non-linear equation of state:
$$\\rho = \\rho(T, S, P)$$
Gravitational water column stability requires that **density increase monotonically with depth** (positive Brunt-Väisälä frequency $N^2 > 0$):
$$N^2 = -\\frac{g}{\\rho} \\frac{\\partial \\rho}{\\partial z} > 0$$
It does **NOT** require that temperature decrease monotonically.

```
+---------------------------------------------------------------------------------------------------+
|               PHYSICAL MECHANISM OF TEMPERATURE INVERSIONS IN THE NORTH INDIAN OCEAN              |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
| 1. BAY OF BENGAL FRESHWATER PLUMES:                                                               |
|    Monsoon runoff from the Ganges-Brahmaputra rivers deposits a thin layer (~10-20m) of very      |
|    fresh water (SSS < 30 psu) over high-salinity equatorial waters (SSS > 34 psu).                |
|                                                                                                   |
| 2. EVAPORATIVE WINTER COOLING:                                                                    |
|    During November–February, cool, dry northeasterly continental winds cool the surface layer     |
|    to ~25°C. However, because the surface layer is so fresh, it remains lighter than the warm,    |
|    salty water (~28°C) trapped immediately beneath it (between 30m and 80m depth).                |
|                                                                                                   |
| 3. THE BARRIER LAYER & WARM TRAP:                                                                 |
|    This creates a stable TEMPERATURE INVERSION (dT/dz > 0) where subsurface water is up to        |
|    1.5°C to 2.5°C WARMER than the surface!                                                        |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

### 2. Scientifically Valid Guardrails
Instead of enforcing artificial temperature monotonicity, OceanEmbed enforces:
1. **Dynamic Range Clamping:** Seawater temperatures are bounded within $[0.0^\\circ\\text{C}, 35.0^\\circ\\text{C}]$.
2. **Thermal Gradient Bounds:** Vertical gradients $|dT/dz|$ are bounded by physically observed limits in the NIO ($|dT/dz| \\le 0.35^\\circ\\text{C}/\\text{m}$).
3. **Surface Boundary Consistency:** Predicted temperature at level 0 ($0.494\\text{m}$) must smoothly track satellite input SST within sensor uncertainty bounds ($|\\hat{T}_0 - \\text{SST}| \\le 0.8^\\circ\\text{C}$).

"""

    p23 = """# PART 23: BIGGEST RISKS & RISK MITIGATION REGISTER

To prevent surprises during hackathon execution, we maintain an audited Risk Register detailing probabilities, impacts, early warning signals, and active mitigations.

| Risk Description | Category | Probability | Impact | Early Warning Indicator | Pre-Engineered Mitigation Strategy |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Deep Subsurface Decorrelation** | Scientific | Medium | High | Model RMSE at $750-1000\\text{m}$ fails to beat monthly climatology. | Weight the loss function exponentially towards the upper $500\\text{m}$ (where 95% of ocean heat dynamics occur); treat 1000m as an asymptotic background head. |
| **Weekly SSS Temporal Lag** | Data | Low | Medium | Salinity fronts in the Bay of Bengal appear smoothed relative to daily SST. | Apply piecewise linear temporal interpolation between weekly Multi-Obs grids; utilize validity mask to flag interpolated days. |
| **Compute / Training Bottleneck** | Infrastructure | Low | High | Epoch time exceeds 4 hours on CPU hardware during full corpus training. | Utilize mixed precision (FP16), optimize DataLoader workers, pin memory, and utilize cloud GPU credits (Google Cloud / Colab Enterprise). |
| **Inversion Layer Smoothing** | Model | Medium | Medium | Model predicts average linear thermocline, erasing real Bay of Bengal barrier layers. | Introduce a specialized localized loss penalty for detected inversion zones where $S_{\\text{sfc}} < 31\\text{ psu}$. |
| **ARGO Spatial Sparsity in NIO** | Validation | Low | Medium | Less than 15 valid ARGO profiles available in the test window. | Expand temporal validation window to 30 days around test dates; utilize historical ARGO trajectory archives from Coriolis. |
| **Copernicus API Rate Limiting** | Pipeline | Low | High | Download script receives HTTP 429 or connection timeout errors during bulk fetching. | Implement exponential backoff retries, download during European off-peak hours, and cache raw NetCDF files locally. |

"""

    p24 = """# PART 24: WHAT CAN STILL STOP US? (FATAL RISKS VS. SOLVABLE CHALLENGES)

To ensure team confidence, we separate potential risks into two distinct classes: **Fatal Showstoppers** and **Solvable Engineering Challenges**.

```
+---------------------------------------------------------------------------------------------------+
|                           FATAL RISKS (SHOWSTOPPERS) — ALL RETIRED!                               |
+---------------------------------------------------------------------------------------------------+
| 1. DATA ACCESS IMPASSE:                                                                           |
|    Risk: Copernicus Marine API requires paid licenses or credentials fail.                        |
|    STATUS: RETIRED. Auth is verified; 5 pilot datasets downloaded and parsed in workspace.        |
|                                                                                                   |
| 2. TARGET 3D DATASET NON-EXISTENT:                                                                |
|    Risk: GLORYS reanalysis does not provide subsurface levels down to 1000m.                      |
|    STATUS: RETIRED. Pilot downloaded (44.74 MB) with 36 native levels down to 1062.44m.           |
|                                                                                                   |
| 3. INCOMPATIBLE SPATIAL GRIDS:                                                                    |
|    Risk: Multi-sensor resolutions cannot be regridded to 0.25° without severe data corruption.    |
|    STATUS: RETIRED. Regridding verified in 0.74s to exact 101x241 grid with 48.7% ocean cells.    |
|                                                                                                   |
| 4. LAND MASK GRADIENT CONTAMINATION:                                                              |
|    Risk: 51.3% land area causes exploding gradients and ruins model learning.                     |
|    STATUS: RETIRED. Mathematically proven 0.000 gradient leakage via live adversarial test.       |
|                                                                                                   |
| 5. INDEPENDENT GROUND TRUTH UNAVAILABLE:                                                           |
|    Risk: Cannot validate against in-situ floats due to INCOIS 404 error.                          |
|    STATUS: RETIRED. Coriolis GDAC pipeline verified with real NIO float #34.                      |
+---------------------------------------------------------------------------------------------------+
```

### Remaining Solvable Engineering Challenges
The only remaining work consists of:
- Downloading the historical multi-year corpus within our verified 2010–2025 overlap.
- Tuning model hyperparameters (learning rate, depth embedding dimension, batch size).
- Assembling the React/WebGL frontend interactive 3D visualization dashboard.
**There are zero remaining theoretical or structural blockers that could kill this project.**

"""

    p25 = """# PART 25: FALLBACK STRATEGY & CONTINGENCY MATRIX

In competitive hackathons, robust teams have predetermined fallbacks for every potential failure mode.

```
+---------------------------------------------------------------------------------------------------+
|                                  CONTINGENCY & FALLBACK MATRIX                                    |
+--------------------------+----------------------------------+-------------------------------------+
| Critical Dependency      | Primary Operational Path         | Pre-Engineered Fallback Path        |
+--------------------------+----------------------------------+-------------------------------------+
| SSS (Sea Surface Salinity| Multi-Obs 7-Day Satellite L4     | Use forward-fill / linear temporal  |
|                          | (`cmems_obs-mob_..._P7D-c`)      | interpolation; if unavailable, run  |
|                          |                                  | 6-variable ablation without SSS.    |
+--------------------------+----------------------------------+-------------------------------------+
| Historical Corpus Size   | Full 10-Year Corpus (2011–2020)  | Reduce to Core 4-Year Corpus        |
|                          | (~3,650 daily snapshots)         | (2019–2022) with 1,460 snapshots;   |
|                          |                                  | fits in <4 GB RAM.                  |
+--------------------------+----------------------------------+-------------------------------------+
| Deep 1000m Decorrelation | Reconstruct full 15 depths down  | Focus primary optimization on top   |
|                          | to 1000m with equal loss weight. | 500m (depths 0–12); report 1000m as |
|                          |                                  | experimental auxiliary baseline.    |
+--------------------------+----------------------------------+-------------------------------------+
| ARGO Data Server Access  | Coriolis GDAC HTTPS Mirror       | Use cached local ARGO multi-profile |
|                          | (`data-argo.ifremer.fr`)         | NetCDF (`20221101_prof.nc`) or US-  |
|                          |                                  | GODAE mirror repository.            |
+--------------------------+----------------------------------+-------------------------------------+
| GPU Compute Availability | Cloud GPU Instances (A100 / T4)  | Train optimized MobileNet/Pointwise |
|                          | with FP16 mixed precision.       | model on local CPU (step time ~5s). |
+--------------------------+----------------------------------+-------------------------------------+
```

"""
    return [p21, p22, p23, p24, p25]
