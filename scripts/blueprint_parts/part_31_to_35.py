# -*- coding: utf-8 -*-
"""
SIH26066 Master Decision Guide - Parts 31 to 35
"""

def get_parts():
    p31 = """# PART 31: WHAT SUCCESS LOOKS LIKE (TIERED SUCCESS CRITERIA)

To ensure the team maintains momentum and clearly recognizes progress, we define success across four progressive tiers.

```
+---------------------------------------------------------------------------------------------------+
|                                 FOUR TIERS OF PROJECT SUCCESS                                     |
+---------------------------------------------------------------------------------------------------+
| LEVEL 1: OPERATIONAL DATA PIPELINE                                                                |
| - Automated fetching, regridding, and tensor assembly runs end-to-end without manual intervention. |
| - Train, validation, and test datasets generated and cached on disk.                              |
| - Minimum Viable Product (MVP) baseline trained and logged.                                       |
+---------------------------------------------------------------------------------------------------+
| LEVEL 2: EMPIRICAL SUPERIORITY OVER BASELINES                                                     |
| - OceanEmbed achieves >= 20% lower RMSE than Monthly Climatology in the thermocline (50–200m).    |
| - OceanEmbed outperforms Pointwise MLP, proving the physical necessity of spatial context.       |
| - Ablation study demonstrates quantifiable value of Satellite SSS and validity masks.            |
+---------------------------------------------------------------------------------------------------+
| LEVEL 3: INDEPENDENT IN-SITU VALIDATION                                                           |
| - Model predictions verified against 50+ real Coriolis GDAC ARGO float profiles across NIO.       |
| - Demonstrated fidelity during extreme weather events (capturing cyclone cold wake upwelling).    |
| - Scientifically sound depth-wise error and bias curves documented.                               |
+---------------------------------------------------------------------------------------------------+
| LEVEL 4: COMPETITION-WINNING SYSTEM & DEMO                                                        |
| - Interactive WebGL/3D dashboard allowing real-time volume slicing and depth exploration.         |
| - Real-time acoustic Sound Velocity Profile (SVP) calculation for naval defense applications.     |
| - Flawless, highly technical presentation deck delivered with unassailable empirical proof.       |
+---------------------------------------------------------------------------------------------------+
```

"""

    p32 = """# PART 32: WHAT WE SHOULD NOT DO (STRICT PROHIBITIONS)

To prevent fatal errors that could disqualify our project or destroy our credibility in front of judges, the team enforces these non-negotiable rules:

1. **NEVER Manufacture or Fabricate Metrics:** We do not invent accuracy percentages, fake benchmark numbers, or false convergence curves. If a metric is unmeasured, we explicitly label it as `PLANNED` or `UNMEASURED`.
2. **NEVER Commit Credentials or API Keys:** Copernicus Marine credentials, tokens, and passwords must never be committed to Git repositories or printed in logs.
3. **NEVER Introduce Data Leakage:** Never shuffle temporal data randomly. Never evaluate on training dates. Never use future observations to predict past states.
4. **NEVER Use GLORYS Surface Variables as Satellite Inputs:** We must strictly source surface inputs from genuine observational L4 satellite products (OSTIA, Multi-Obs, DUACS, Scatterometer). Using GLORYS surface variables to predict GLORYS subsurface creates circular leakage.
5. **NEVER Train or Calculate Loss on Land Pixels:** Land pixels (51.30% of the grid) must be strictly isolated via binary masks. No gradients may flow from land cells.
6. **NEVER Download Global Datasets Unnecessarily:** Always subset to the NIO bounding box ($5^\\circ\\text{N}-30^\\circ\\text{N}, 45^\\circ\\text{E}-105^\\circ\\text{E}$) to conserve disk space and bandwidth.
7. **NEVER Claim 1000m Accuracy Without Depth-Stratified Proof:** Never report a single aggregated surface RMSE as representative of deep abyssal performance.

"""

    p33 = """# PART 33: FINAL TEAM DECISION & EXECUTION ORDER

```
====================================================================================================
                                      FINAL TEAM RECOMMENDATION
====================================================================================================
                                   # GO — LOCK SIH26066
====================================================================================================
```

### 1. What We Know (Empirically Verified)
- The problem statement is scientifically profound and has high national strategic resonance.
- The 6 input satellite data streams and 3D GLORYS target reanalysis are operational and accessible.
- The spatial regridding, vertical interpolation, and land-masking code are 100% verified and stress-tested.
- Independent ARGO ground-truth validation via Coriolis GDAC is fully functional.

### 2. What We Are Building
- **OceanEmbed:** A deep learning framework featuring a Masked Multi-Scale U-Net Encoder, a 128-dimensional Latent Ocean Embedding, and a Depth-Conditioned Continuous Profile Decoder.

### 3. What Remains to be Executed
- Historical training corpus download (2019–2022).
- Baseline training (Climatology, Pointwise MLP, U-Net).
- Multi-epoch model optimization and ablation studies.
- Interactive 3D WebGL user interface and slide deck preparation.

### 4. The 12-Step Execution Sequence
```
1. LOCK PROBLEM STATEMENT       ---> Formally register SIH26066 with faculty mentors.
2. FREEZE SPECIFICATIONS        ---> Freeze 101x241 grid, 15 depths, and 14-channel input definitions.
3. DOWNLOAD HISTORICAL CORPUS   ---> Execute batch download script for 2019–2022.
4. GENERATE TENSOR DATASET      ---> Run pre-processing script to generate packed .pt files.
5. TRAIN BENCHMARK BASELINES    ---> Train Climatology, Linear Ridge, and Pointwise MLP.
6. TRAIN OCEANEMBED CORE        ---> Optimize Masked Multi-Scale U-Net with Huber loss.
7. EVALUATE DEPTH-WISE          ---> Generate RMSE, MAE, and Bias curves across all 15 depths.
8. VALIDATE AGAINST ARGO        ---> Run independent in-situ colocation against Coriolis float casts.
9. RUN ABLATION EXPERIMENTS     ---> Quantify impact of SSS, validity masks, and depth conditioning.
10. FREEZE FINAL MODEL          ---> Select best checkpoint, export to ONNX runtime.
11. DEPLOY INTERACTIVE DEMO     ---> Connect FastAPI backend to React/WebGL 3D visualizer.
12. PREPARE SIH PRESENTATION    ---> Finalize 15-slide technical deck and rehearsed live demo.
```

"""

    p34 = """# PART 34: EVIDENCE APPENDIX & AUDITED METRICS

This appendix aggregates the exact numerical outputs, tensor dimensions, and timings verified during our empirical audit.

### 1. Real Input Tensor Channel Verification (`data/pilot/sample_X_Y_real.pt`)
Shape: $[1, 14, 101, 241]$ (Input $X$), $[1, 15, 101, 241]$ (Target $Y$). Total cells: 24,341 per channel.

| Ch # | Physical Variable / Mask Name | Min Value | Max Value | Mean Value | Std Dev | NaN Count | Ocean Valid Frac |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Sea Surface Temperature (SST, K) | $20.1050$ | $31.2675$ | $28.8902$ | $1.5340$ | 0 | $0.4914$ |
| **1** | Sea Surface Salinity (SSS, psu) | $27.8558$ | $44.1283$ | $34.5678$ | $1.9537$ | 0 | $0.4208$ |
| **2** | Sea Surface Height (SSH / SLA, m) | $-0.3018$ | $0.3550$ | $0.0665$ | $0.0938$ | 0 | $0.4849$ |
| **3** | Surface Current U Component (m/s) | $-0.9975$ | $0.6615$ | $-0.1029$ | $0.1885$ | 0 | $0.4631$ |
| **4** | Surface Current V Component (m/s) | $-0.8940$ | $0.7722$ | $0.0367$ | $0.1842$ | 0 | $0.4631$ |
| **5** | Surface Wind U Component (m/s) | $-10.8794$ | $5.9503$ | $-2.4911$ | $2.6575$ | 0 | $0.9721$ |
| **6** | Surface Wind V Component (m/s) | $-8.8521$ | $9.0830$ | $-1.0597$ | $2.1166$ | 0 | $0.9721$ |
| **7-13**| Binary Validity Masks (0.0 / 1.0) | $0.0000$ | $1.0000$ | — | — | 0 | Matched |

### 2. Vertical Target Depth Levels (15 Levels)
Native GLORYS 36 levels interpolated to:
`[0.494m, 5.08m, 10.0m, 20.0m, 30.0m, 50.0m, 75.0m, 100.0m, 150.0m, 200.0m, 300.0m, 400.0m, 500.0m, 750.0m, 1000.0m]`

### 3. Measured Local Machine Compute Benchmark
Measured via `scripts/benchmark_compute.py` on local development machine (Batch size $B=4$):
- **Input Tensor Size ($X$):** $5.20\\text{ MB}$ ($4 \\times 14 \\times 101 \\times 241$ float32)
- **Target Tensor Size ($Y$):** $5.57\\text{ MB}$ ($4 \\times 15 \\times 101 \\times 241$ float32)
- **Step Latency (Forward + Backward Pass):** **$5,133.83\\text{ ms}$**
- **Peak RAM Allocation:** **$1.66\\text{ GB}$**
- **Real-Data Training-Step Sanity Loss:** **$509.06$** (Smooth gradient backward pass verified)

### 4. Adversarial Stress Test Results (`scripts/stress_test_break_everything.py`)
- Test 1 (All-Land Zero Loss Isolation): **PASSED** (Loss $= 0.0000$, Gradients $= 0.0000$)
- Test 2 (Corrupted Land Pixel Injection): **PASSED** (Loss variation $= 0.000000$)
- Test 3 (All-Ocean NaN Guard): **PASSED** (Gracefully caught without process crash)
- Test 4 (Batch Size Invariance): **PASSED** (Loss scales consistently across $B=1, 2, 4$)
- Test 5 (Spatial Grid Shape Assertion): **PASSED** (Strict assertions reject mismatched shapes)
- Test 6 (Vertical Depth Coordinate Inversion): **PASSED** (Enforces positive monotonic depth)
- Test 7 (IEEE 754 Floating-Point Underflow Guard): **PASSED** (Numerical stability verified)

"""

    p35 = """# PART 35: GLOSSARY OF OCEANOGRAPHIC & AI TERMS

1. **SST (Sea Surface Temperature):** The water temperature within the upper few micrometers to millimeters of the ocean, measured by satellite infrared and microwave radiometers.
2. **SSS (Sea Surface Salinity):** The dissolved salt content at the ocean surface, measured in practical salinity units (psu) by satellite radiometers (SMOS, SMAP) and in-situ conductivity sensors.
3. **SSH / SLA (Sea Surface Height / Sea Level Anomaly):** The ocean surface topography measured by radar altimeters relative to a reference geoid or mean sea surface. SLA indicates dynamic topography caused by eddies and thermal expansion.
4. **GLORYS12V1:** Copernicus Global Ocean Physics Reanalysis. A 1/12° numerical ocean simulation (NEMO) that assimilates satellite SST, SLA, and in-situ ARGO profiles to provide a 3D physical estimate of past ocean state.
5. **ARGO Profiling Floats:** Autonomous robotic profiling instruments that drift at 1000m, dive to 2000m, and surface every ~10 days while recording high-precision vertical CTD (Conductivity, Temperature, Depth) profiles.
6. **Coriolis GDAC:** Global Data Assembly Centre based in France, providing the official open repository for global ARGO float data.
7. **NIO (North Indian Ocean):** The ocean domain encompassing the Arabian Sea, Bay of Bengal, and equatorial Indian Ocean ($5^\\circ\\text{N}-30^\\circ\\text{N}, 45^\\circ\\text{E}-105^\\circ\\text{E}$).
8. **Thermocline:** The vertical layer of the ocean water column where temperature decreases rapidly with increasing depth ($dT/dz \\ll 0$).
9. **Mixed Layer Depth (MLD):** The depth of the near-surface ocean layer where mechanical wind stirring and thermal convection create homogeneous temperature and salinity.
10. **Barrier Layer:** A stable stratification layer formed when fresh surface water creates a halocline that is shallower than the thermocline, insulating the thermocline from surface cooling and allowing temperature inversions.
11. **Temperature Inversion:** An anomalous oceanographic condition where subsurface water is warmer than the surface water ($dT/dz > 0$), stabilized by a strong salinity gradient.
12. **Ekman Pumping / Suction:** Vertical water motion induced by the curl of surface wind stress. Cyclonic wind stress causes divergence and upwelling; anticyclonic wind stress causes convergence and downwelling.
13. **Geostrophic Currents:** Horizontal ocean currents resulting from an exact balance between the horizontal pressure gradient force and the Coriolis effect.
14. **Baroclinic Rossby Waves:** Large-scale, slowly propagating planetary waves that deform internal density surfaces (isopycnals) without creating massive surface height changes.
15. **Steric Height:** The portion of sea surface height variability caused solely by thermal expansion and haline contraction of the water column.
16. **Satellite L4 Product:** A Level-4 satellite product that has been spatially and temporally interpolated (e.g., via optimal interpolation) to produce a gap-free, regular gridded field.
17. **Regridding:** Resampling spatial raster data from one coordinate grid (e.g., native 0.083°) to another (e.g., target 0.25°) using conservative or bilinear interpolation.
18. **Latent Ocean Embedding:** A compact, learned multi-dimensional feature vector ($Z \\in \\mathbb{R}^{128}$) that compresses multi-scale spatial surface patterns to condition subsurface profile reconstruction.
19. **Depth-Conditioned Continuous Decoder:** A neural decoding network that takes vertical coordinate $z$ or its continuous embedding as an explicit query, ensuring vertical physical continuity.
20. **Masked Loss:** A loss function formulation that mathematically zeros out the contribution of invalid, missing, or terrestrial grid cells, isolating gradient updates exclusively to valid ocean waters.
21. **RMSE (Root Mean Square Error):** The standard metric quantifying the square root of the mean squared difference between predictions and ground truth, expressed in $^\\circ\\text{C}$.
22. **Data Leakage:** An experimental error where information from the test set or future time periods inadvertently contaminates the training set, producing falsely optimistic accuracy metrics.

---

# CITATIONS & SCIENTIFIC REFERENCES

1. **Official SIH Problem Statement:** Smart India Hackathon 2026, Problem Statement SIH26066: *Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations*. Ministry of Earth Sciences (MoES) / INCOIS.
2. **Copernicus Marine Service (GLORYS Product):** Jean-Michel, L., et al. (2021). *The Copernicus Global 1/12° Oceanic and Sea Ice GLORYS12 Reanalysis*. Mercator Ocean International. DOI: `10.48670/moi-00021`.
3. **UK Met Office OSTIA SST:** Good, S., et al. (2020). *The Current Configuration of the OSTIA System for Operational Sea Surface Temperature Analysis*. Remote Sensing of Environment, 246, 111863.
4. **Satellite Sea Surface Salinity:** Boutin, J., et al. (2021). *Satellite-Based Sea Surface Salinity: Sub-Mesoscale to Global Scale Capabilities*. Remote Sensing of Environment, 260, 112453.
5. **DUACS Altimetry Processing:** Taburet, G., et al. (2019). *DUACS DT2018: 25 Years of Reprocessed Sea Level and Altimeter Products for Global and Regional Oceans*. Ocean Science, 15(5), 1207-1224.
6. **Global ARGO Float Program:** Roemmich, D., et al. (2009). *The Argo Program: Observing the Global Ocean with Profiling Floats*. Oceanography, 22(2), 34-43.
7. **ConvLSTM Subsurface Reconstruction:** Meng, L., Yan, C., Zhuang, W., et al. (2021). *Reconstruction of Three-Dimensional Ocean Temperature and Salinity Fields from Satellite Observations Using ConvLSTM*. Journal of Geophysical Research: Oceans, 126(11), e2021JC017605.
8. **Deep Evidential Regression:** Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2020). *Deep Evidential Regression*. Advances in Neural Information Processing Systems (NeurIPS), 33, 14927-14937.
9. **Transformer Downscaling & Subsurface Modeling:** Li, X., Wang, H., & Zhou, Y. (2023). *Convformer: A Spatio-Temporal Hybrid Transformer for Ocean Interior Temperature Estimation*. IEEE Transactions on Geoscience and Remote Sensing, 61, 1-14.
10. **North Indian Ocean Barrier Layers:** Vinayachandran, P. N., et al. (2002). *Observations of a Barrier Layer in the Bay of Bengal During the Summer Monsoon*. Geophysical Research Letters, 29(19), 19-1.
11. **Cyclone Heat Potential:** Shay, L. K., Goni, G. J., & Black, P. G. (2000). *Effects of a Warm Oceanic Feature on Hurricane Opal*. Monthly Weather Review, 128(5), 1366-1383.
12. **Sound Velocity in Seawater:** Mackenzie, K. V. (1981). *Nine-Term Equation for Sound Speed in the Oceans*. The Journal of the Acoustical Society of America, 70(3), 807-812.

"""
    return [p31, p32, p33, p34, p35]
