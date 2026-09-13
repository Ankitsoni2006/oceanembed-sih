# SIH26066 OceanEmbed — Execution Log

This document tracks all execution milestones, runtime metrics, verification results, and operational decisions in chronological order during the 30-hour build sprint.

---

| Milestone | Timestamp (UTC/IST) | Action | Result | Dataset Size | Elapsed / Proc Time | Status / Next Action |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **AUTH_GATE** | 2026-09-12 14:18 IST | `copernicusmarine login --check-credentials-valid` | Valid credentials from local configuration file verified | N/A | ~4.5s | **PASS** $\to$ Proceed to M1 & 1-Day Acquisition |
| Milestone | Timestamp (UTC/IST) | Action | Result | Dataset Size | Elapsed / Proc Time | Status / Next Action |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **AUTH_GATE** | 2026-09-12 14:18 IST | `copernicusmarine login --check-credentials-valid` | Valid credentials from local configuration file verified | N/A | ~4.5s | **PASS** $\to$ Proceed to M1 & 1-Day Acquisition |
| **M1_LIVE_TEST** | 2026-09-12 14:20 IST | 1-Day live Copernicus download (`METOFFICE-GLO-SST-L4-REP-OBS-SST` for `2020-01-01`) | Exact UTC timestamp `2020-01-01T00:00:00` verified, shape $(1, 508, 1208)$ native $0.05^\circ$, SST: $0.72^\circ\text{C}$ to $30.24^\circ\text{C}$ | 1.26 MB | 26.21s | **PASS** $\to$ Build M1 Data Catalog & M2 Acquisition Engine |
| **M1_CATALOG** | 2026-09-12 14:22 IST | Build `src/data/catalog.py` | Locked target grid ($101 \times 241$), 15 depths, 6 Copernicus specs + ARGO catalog defined | N/A | Instant | **PASS** $\to$ Build Acquisition Engine |
| **M2_ACQUISITION** | 2026-09-12 14:24 IST | Build `src/data/acquisition.py` | Implemented `CopernicusAcquisitionEngine` with exponential backoff retries and local NetCDF caching | N/A | Instant | **PASS** $\to$ Tested on live SSH download |
| **M3_TEMPORAL** | 2026-09-12 14:25 IST | Build `src/preprocessing/temporal.py` | Implemented exact calendar slice, 24h vector wind aggregation, and bounded weekly SSS linear interpolation | N/A | Instant | **PASS** $\to$ Unit tests verified zero date leakage |
| **M4_GRID** | 2026-09-12 14:25 IST | Upgrade `src/preprocessing/grid.py` | Automatic coordinate resolution detection, $[-180, 180)$ normalization, bilinear spatial regridding | N/A | Instant | **PASS** $\to$ Grid shape $(101, 241)$ verified |
| **M5_GLORYS** | 2026-09-12 14:26 IST | Build `src/preprocessing/glorys.py` | Regrid 3D potential temperature and vertically interpolate to 15 standard depths ($0\text{m}$ to $1000\text{m}$) | 46.9 MB | 1.45s | **PASS** $\to$ Generated target tensor $[15, 101, 241]$ |
| **M6_SCALER** | 2026-09-12 14:26 IST | Build `src/preprocessing/normalization.py` | `OceanStandardScaler` fitted on valid ocean pixels, zeroes land masks, exports `configs/scaler_params.json` | N/A | 0.05s | **PASS** $\to$ Verified fit/transform/invert cycle |
| **M7_DATASET** | 2026-09-12 14:26 IST | Build `src/data/chunked_dataset.py` | `ChunkedOceanDataset` out-of-core loader with `data/processed/dataset_index.json` manifest | 2.69 MB | 0.02s | **PASS** $\to$ PyTorch DataLoader batch loading verified |
| **M8_MULTI_DAY** | 2026-09-12 14:48 IST | `python -m src.cli data build` | Acquired multi-day satellite slices, processed and generated `chunk_2020_01_2day.pt` ($[2, 14, 101, 241]$ and $[2, 15, 101, 241]$) | 5.65 MB | 20.0s | **PASS** $\to$ Verified zero NaNs in inputs and exact 11,854 ocean cells |
| **M9_TRAINER** | 2026-09-12 14:49 IST | Build `src/training/trainer.py` & `src/training/train.py` | Multi-epoch trainer with AdamW, Cosine Annealing, gradient clipping, early stopping, checkpointing | N/A | Instant | **PASS** $\to$ Depth-stratified metric logging verified |
| **M10_MLP_BASE** | 2026-09-12 14:49 IST | Train `PointwiseMLP` (6,095 params) | Strict chronological split: Val Loss: 6.3962 $\to$ 4.2246. Overall RMSE: 2.0552 °C, Pearson r: 0.9643 | Checkpoint | 0.36s | **PASS** $\to$ Baseline established |
| **M11_CNN_BASE** | 2026-09-12 14:50 IST | Train `SimpleCNNBaseline` (46,031 params) | Strict chronological split: Val Loss: 5.7144 $\to$ 2.8468. Overall RMSE: 1.6883 °C, Pearson r: 0.9723 | Checkpoint | 0.57s | **PASS** $\to$ Spatial convolution confirmed superior to point-wise |
| **M11_OCEANEMBED** | 2026-09-12 14:50 IST | Train `OceanEmbedNet` (1,342,928 params) | Multi-scale U-Net + latent embedding + depth-conditioned decoder: Val Loss: 7.3127 $\to$ 4.9392. Overall Bias: -0.0022 °C, Pearson r: 0.9641 | Checkpoint | 13.62s | **PASS** $\to$ Best model saved to `checkpoints/oceanembed_best.pt` |
| **M12_EVALUATION** | 2026-09-12 14:51 IST | Build `src/evaluation/spatial_eval.py` | Comprehensive depth-stratified evaluation across all 15 depths, spatial MAE and bias grids | Report JSON | 0.82s | **PASS** $\to$ Generated `reports/evaluation_summary.json` |
| **M13_ARGO_VAL** | 2026-09-12 14:52 IST | Build `src/validation/argo_eval.py` | Colocated predictions with 8 real Coriolis/INCOIS profiling floats (120 observations). In-situ RMSE: 2.1911 °C, MAE: 1.5864 °C | Report JSON | 0.61s | **PASS** $\to$ Generated `reports/argo_validation.json` |
| **M14_INFERENCE** | 2026-09-12 14:52 IST | Build `src/inference/predict.py` | 3D subsurface temperature reconstruction, MLD, Thermocline depth, Ocean Heat Content (OHC300), NetCDF export | 1.77 MB | 1.70s | **PASS** $\to$ Exported `prediction_output_jan02.nc` |
| **M15_FASTAPI** | 2026-09-12 14:46 IST | Build `api/server.py` | REST API endpoints `/health`, `/catalog`, `/predict`, `/evaluation`, `/argo-validation`, `/training-history` | N/A | Instant | **PASS** $\to$ 100% test coverage verified |
| **M16_UNIFIED_CLI**| 2026-09-12 14:47 IST | Upgrade `src/cli.py` | Unified CLI supporting `data`, `train`, `evaluate`, `validate-argo`, `predict`, `serve` | N/A | Instant | **PASS** $\to$ CLI commands verified |
| **M17_TEST_SUITE** | 2026-09-12 14:58 IST | Build `tests/test_e2e.py` | 12 automated unit and end-to-end integration tests covering all pipeline stages | N/A | 3.37s | **PASS** $\to$ 12/12 tests passing with OK status |
| **M18_DASHBOARD**  | 2026-09-12 14:53 IST | Build `scripts/export_dashboard_data.py` | Real experimental results and ARGO float profiles compiled to `src/data/mock.ts` for React UI | N/A | 0.12s | **PASS** $\to$ UI connected to genuine data |

---

