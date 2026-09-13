# OceanEmbed — Current Development Handoff

## Completed
- Defined project directories (`data/`, `src/`, `configs/`, `tests/`, `docs/`).
- Formalized central configuration for NIO grid bounds and target resolutions (`configs/config.py`).
- Formalized surface dataset metadata configuration (`configs/surface_inputs.py`).
- Implemented the OceanEmbed core PyTorch architecture (Masked Multi-Scale U-Net -> Latent Ocean Embedding -> Depth Decoder).
- Implemented Baselines (Point-wise MLP, Simple CNN).
- Built masking logic to safely convert 7 raw features into a 14-channel tensor (`src/preprocessing/masks.py`).
- Built a masked loss function and evaluation metric suite that correctly skips unobserved oceanic regions and land (`src/training/loss.py`, `src/evaluation/metrics.py`).
- Set up unit tests focusing on temporal leakage and mask mechanics (`tests/test_leakage.py`).

## Tested Successfully
- Unit tests for mask generation, temporal train/val splitting logic, and tensor shapes execute and pass.
- Model mechanics (channels, depth conditioning logic) are structurally sound in pure PyTorch.

## Not Yet Tested
- E2E training loop. (Awaits real data).
- Regridding logic on live xarray DataSets.

## Blocked by GLORYS Audit
- Exact dimensions, native coordinates, missing-data value conventions (`NaN` vs `-32767`), and depth indexing in the NetCDF files.
- Actual construction of the PyTorch DataLoaders requires the real GLORYS target tensors.

## Blocked by Surface Data
- Final historical timeline selection. (We have mapped 2016-2020 as a theoretical safe training zone, but require testing all 7 live satellite API streams for this period to verify availability).

## Blocked by ARGO
- `ArgoColocator` in `src/validation/argo.py` is currently a skeletal interface. We need the specific output format from the INCOIS Live Access Server to map float coordinates accurately to our 0.25° grid.

## Scientific Questions Remaining
- Should the physical loss constraint (e.g. limiting severe unphysical temperature inversions) be added? (Pending analysis of real GLORYS profiles).
- What is the exact padding requirement for the `101x241` spatial grid inside the U-Net? (Current PyTorch implementation uses dynamic `F.pad` during the decoder upsampling to maintain strict dimensionality regardless of pooling artifacts).

## Exact Next Steps After GLORYS Audit
1. Run the Python `copernicusmarine` CLI subset download script against the verified `cmems_mod_glo_phy_my_0.083deg_P1D-m` GLORYS dataset.
2. Load a 1-day GLORYS subset via `xarray`, regrid it using `src/preprocessing/grid.py`, and extract the exact 15 depths requested by SIH.
3. Validate that the GLORYS land-mask accurately aligns with the mathematical 5°N–30°N, 45°E–105°E grid configuration.
4. Pass the verified shape and mask to the `OceanDataset` to initialize full training.
