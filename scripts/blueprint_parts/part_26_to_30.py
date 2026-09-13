# -*- coding: utf-8 -*-
"""
SIH26066 Master Decision Guide - Parts 26 to 30
"""

def get_parts():
    p26 = """# PART 26: SIX-PERSON TEAM EXECUTION PLAN & ROLE ALLOCATION

Winning the Smart India Hackathon requires clear operational boundaries, zero duplicated effort, and daily cross-functional integration across all six members.

```
+---------------------------------------------------------------------------------------------------+
|                                SIX-PERSON COLLABORATIVE STRUCTURE                                 |
+---------------------------------------------------------------------------------------------------+
| MEMBER 1: Data Engineering Lead                 MEMBER 2: Oceanographic Physics Lead              |
| - Copernicus bulk downloading & caching         - Regridding quality & vertical interpolation     |
| - NetCDF parser & temporal alignment            - Thermocline physics & barrier layer audit       |
| - PyTorch Dataset & DataLoader optimization     - Input normalization & anomaly baselines         |
|                          \\                             /                                          |
|                           v                           v                                           |
|                  [SHARED INTEGRATION GATE 1: HARMONIZED 14-CH CORPUS]                             |
|                                         |                                                         |
|                          +--------------+--------------+                                          |
|                          |                             |                                          |
|                          v                             v                                          |
| MEMBER 3: ML Architecture Lead                  MEMBER 4: Training & Optimization Lead            |
| - Masked Multi-Scale U-Net Encoder              - AdamW, cosine annealing, mixed precision        |
| - 128-dim Latent Ocean Embedding module         - Masked Huber loss + gradient regularization     |
| - Continuous Depth-Conditioned Decoder          - Ablation studies (no-mask, no-SSS, single-scale)|
|                          \\                             /                                          |
|                           v                           v                                           |
|                  [SHARED INTEGRATION GATE 2: CONVERGED OCEANEMBED MODEL]                          |
|                                         |                                                         |
|                          +--------------+--------------+                                          |
|                          |                             |                                          |
|                          v                             v                                          |
| MEMBER 5: ARGO & Scientific Validation Lead     MEMBER 6: Frontend & Systems Integration Lead     |
| - Coriolis GDAC multi-profile colocation        - Interactive 3D WebGL / Three.js ocean dashboard |
| - Depth-by-depth RMSE, Bias, Correlation        - Real-time slice inspector (lon, lat, depth)     |
| - Extreme cyclone cold wake case studies        - Production FastAPI model serving backend        |
+---------------------------------------------------------------------------------------------------+
```

### 1. Detailed Role Allocation & Concrete Deliverables

| Team Member & Role | Primary Responsibilities | Key Concrete Deliverable | Critical Ownership |
| :--- | :--- | :--- | :--- |
| **Member 1**<br>Data Engineering Lead | Copernicus Marine bulk downloading, local caching, NetCDF parsing, PyTorch DataLoader optimization with pinned memory. | High-speed, leak-free DataLoader yielding $(X, Y)$ batches in $<50\\text{ms}$. | Data Pipeline & Ingestion |
| **Member 2**<br>Oceanographic Physics Lead | Regridding validation, PCHIP vertical interpolation audit, NIO barrier layer identification, physical range checking. | Validation report certifying that regridded fields match published climatology. | Geophysical Correctness |
| **Member 3**<br>ML Architecture Lead | Implementation of Masked Multi-Scale U-Net, 128-dim Latent Embedding, Depth Decoder, and 4 baseline architectures. | Modular `src/models/` package passing unit tests for gradients and dimensions. | Model Topology |
| **Member 4**<br>Training & Evaluation Lead | Multi-epoch optimization, AdamW schedules, mixed precision FP16, loss ablation studies, checkpoint serialization. | Model training curves, evaluation metrics logs, and loss ablation tables. | Convergence & Metrics |
| **Member 5**<br>ARGO Validation Lead | Coriolis GDAC pipeline, in-situ float colocation, vertical depth-error profiling, cyclone cold wake case studies. | Definitive independent ARGO validation report and scientific error curves. | Scientific Defensibility |
| **Member 6**<br>Frontend & Integration Lead | Interactive 3D WebGL / deck.gl dashboard, depth slider ($0-1000\\text{m}$), FastAPI inference service, slide deck. | Operational, responsive 3D ocean visualizer ready for live jury demonstration. | Presentation & UI/UX |

"""

    p27 = """# PART 27: PHASE-BY-PHASE EXECUTION ROADMAP

The project is structured into 12 disciplined phases with explicit inputs, outputs, owners, and definitions of done.

```
+---------------------------------------------------------------------------------------------------+
|                                  12-PHASE EXECUTION ROADMAP                                       |
+-------+-----------------------------+---------------------+-------------------+-------------------+
| Phase | Phase Name & Focus          | Estimated Effort    | Primary Owner     | Target Milestone  |
+-------+-----------------------------+---------------------+-------------------+-------------------+
| P0    | Problem Statement Lock      | Completed (Audit)   | Entire Team       | Gate 0 Passed     |
| P1    | Historical Corpus Download  | 2 Days              | Member 1          | 4-Year Core NetCDF|
| P2    | Dataset Harmonization       | 1 Day               | Member 1, 2       | Regridded Grids   |
| P3    | Tensor Dataset Generation   | 1 Day               | Member 1          | Packed .pt Files  |
| P4    | Baseline Suite Training     | 1 Day               | Member 3, 4       | Climatology & MLP |
| P5    | OceanEmbed v1 Core Training | 2 Days              | Member 3, 4       | Masked U-Net Model|
| P6    | Ablation & Feature Studies  | 1 Day               | Member 4          | Component Proofs  |
| P7    | ARGO Independent Validation | 1 Day               | Member 5          | In-situ Float Test|
| P8    | Cyclone Case Studies        | 1 Day               | Member 2, 5       | Tauktae & Amphan  |
| P9    | Inference Optimization      | 1 Day               | Member 3, 6       | ONNX / TensorRT   |
| P10   | 3D Interactive Web Demo     | 2 Days              | Member 6          | Live WebGL UI     |
| P11   | Final SIH Pitch & Slide Deck| 1 Day               | Entire Team       | Final Competition |
+-------+-----------------------------+---------------------+-------------------+-------------------+
```

### Detailed Phase Specifications
- **Phase 0 (Problem Lock):** COMPLETED. Full data feasibility audit executed; SSS pilot confirmed; land-mask math proven; 7 stress tests passing.
- **Phase 1 (Historical Download):** Fetch 4 years (2019–2022) of daily SST, SSS, SSH, Currents, Winds, and 3D GLORYS for the NIO bounding box ($5^\\circ\\text{N}-30^\\circ\\text{N}, 45^\\circ\\text{E}-105^\\circ\\text{E}$).
- **Phase 2 (Harmonization):** Batch regrid all variables to $0.25^\\circ$ ($101 \\times 241$), interpolate SSS weekly-to-daily, average hourly winds to daily vectors, and vertically interpolate GLORYS to 15 depths.
- **Phase 3 (Tensor Generation):** Compute binary masks, apply z-score normalization, assemble 14-channel input tensors and 15-channel target tensors, and export train/val/test splits.
- **Phase 4 (Baselines):** Train Monthly Climatology, Pointwise MLP, and Standard 2D U-Net baselines. Log baseline RMSE per depth.
- **Phase 5 (OceanEmbed v1):** Train the Masked Multi-Scale U-Net with Latent Ocean Embedding and Depth-Conditioned Decoder. Optimize with Masked Huber loss.
- **Phase 6 (Ablations):** Evaluate model variants: (a) without validity masks, (b) without SSS, (c) without multi-scale receptive fields, (d) without depth conditioning.
- **Phase 7 (ARGO Validation):** Download all NIO ARGO float profiles for test years 2023–2024 from Coriolis GDAC. Run colocation harness and compute in-situ depth-wise RMSE.
- **Phase 8 (Cyclone Studies):** Run inference across historical cyclone events (Cyclone Tauktae, May 2021; Cyclone Amphan, May 2020). Plot subsurface cold wake upwelling cross-sections.
- **Phase 9 (Optimization):** Export PyTorch model to ONNX runtime format. Optimize inference latency to $<100\\text{ms}$ per 3D field on CPU.
- **Phase 10 (Interactive UI):** Build React + deck.gl web application featuring 3D volumetric rendering, depth sliders ($0-1000\\text{m}$), and ARGO float comparison popups.
- **Phase 11 (Final Presentation):** Assemble 15-slide technical pitch deck with live demo fallback video and audited evidence register.

"""

    p28 = """# PART 28: DEFINITION OF DONE & QUALITY GATES

To maintain rigorous software and scientific standards, no phase is considered complete until it satisfies hard, measurable criteria.

```
+---------------------------------------------------------------------------------------------------+
|                                HARD DEFINITION OF DONE GATES                                      |
+---------------------------------------------------------------------------------------------------+
| GATE 1: DATA PIPELINE CERTIFICATION                                                              |
| [ ] All 7 surface variables downloaded for the target date range without corrupted NetCDF headers |
| [ ] Regridding to exactly 101x241 grid verified with zero spatial coordinate inversions           |
| [ ] Land cells (12,487 cells) verified to have Mask=0.0 and Value=0.0                             |
| [ ] Zero NaN or infinite values in packed PyTorch tensors                                         |
+---------------------------------------------------------------------------------------------------+
| GATE 2: BASELINE BENCHMARK ESTABLISHMENT                                                          |
| [ ] Monthly Climatology RMSE computed and tabulated across all 15 depth levels                   |
| [ ] Pointwise MLP trained to convergence on identical train split                                 |
| [ ] Baseline results saved as persistent reference JSON/CSV                                       |
+---------------------------------------------------------------------------------------------------+
| GATE 3: MODEL TRAINING INTEGRITY                                                                  |
| [ ] Train loss and validation loss decrease smoothly without divergence                           |
| [ ] Gradient norms remain bounded (< 5.0) via gradient clipping                                   |
| [ ] Model checkpoint with best validation loss automatically serialized                           |
| [ ] Model outperforms Monthly Climatology by >= 20% in the thermocline (50–200m)                  |
+---------------------------------------------------------------------------------------------------+
| GATE 4: INDEPENDENT IN-SITU VALIDATION                                                            |
| [ ] ARGO colocation executed on at least 50 independent NIO float profiles                        |
| [ ] In-situ depth RMSE and bias plotted alongside GLORYS reanalysis test error                    |
| [ ] Taylor diagram generated showing correlation and normalized standard deviation                |
+---------------------------------------------------------------------------------------------------+
| GATE 5: SYSTEM INTEGRATION & DEMO READINESS                                                       |
| [ ] FastAPI backend serves 3D predictions in < 150ms per request                                  |
| [ ] Interactive UI allows continuous depth scrubbing from 0m to 1000m                             |
| [ ] ARGO float markers clickable in UI with side-by-side profile comparison                       |
+---------------------------------------------------------------------------------------------------+
```

"""

    p29 = """# PART 29: IS IT ACTUALLY POSSIBLE? (EMPIRICAL VERDICT)

### 1. The Short Answer: YES.
The engineering feasibility of this project is no longer an open question. It is an **experimentally demonstrated fact**.

### 2. Why We Are Certain
1. **The Code Already Runs:** In our local workspace, we have already executed the complete chain: downloading 3D GLORYS NetCDFs, regridding OSTIA SST, interpolating satellite SSS, packing the 14-channel input tensor, isolating the 12,487 land cells, executing forward-backward PyTorch training steps, and colocating real Coriolis ARGO floats.
2. **The Physics is Sound:** The relationship between surface dynamic height, wind stress curl, and thermocline depth is well-grounded in geophysical fluid dynamics (Gill, 1982; Chelton et al., 2001).
3. **The Data Exists and is Free:** All required data streams are operational, publicly accessible, and legally authorized for research use.

We are not attempting an unproven scientific breakthrough; we are implementing a rigorous, multi-modal deep learning architecture on verified operational satellite data.

"""

    p30 = """# PART 30: IS THE EFFORT WORTH IT? (RETURN ON INVESTMENT)

For our 6-person team, allocating weeks of intensive engineering effort to SIH26066 yields an extraordinary return across three dimensions:

### 1. High Probability of Winning SIH
- In a competition dominated by shallow web wrappers and generic chatbots, a team presenting a functioning 3D oceanographic deep learning system with verified satellite data and naval acoustic duct applications stands in a league of its own.
- Juries from the Ministry of Earth Sciences, INCOIS, and DRDO/Naval Research will immediately recognize the technical depth and practical utility of this project.

### 2. Genuine Intellectual & Engineering Growth
- Every team member will master advanced geospatial data engineering (NetCDF-4, CF conventions, xarray, Dask), geophysical fluid dynamics, multi-scale deep learning architectures, and high-performance WebGL visualization.
- These skills are directly transferable to elite careers in AI, climate tech, aerospace, and remote sensing.

### 3. Immediate Post-Hackathon Viability
- INCOIS actively funds research and operational projects in ocean modeling. A successful SIH demonstration creates direct pathways for research grants, government incubation, and published peer-reviewed papers.

"""
    return [p26, p27, p28, p29, p30]
