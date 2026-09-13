# SIH26066 — EXECUTION ROADMAP (POST-AUDIT)

## PHASE 0: Feasibility & Data Harmonization Gate (COMPLETED & PASSED)
*   **Deliverable:** Regrid 1 day of GLORYS and 1 day of satellite SST to 0.25° grid, verify 15 depths bracketing, test ARGO colocation, and verify PyTorch DataLoader and OceanEmbed.
*   **Result:** **FULL PASS**. Dimensions `(101, 241)` verified for all inputs/targets. Model loss and backward step verified. ARGO float colocation operational.

## PHASE 1: Authentication & Data Acquisition (Immediate Next Step)
*   **Action:** Run `copernicusmarine login` in local terminal to store authentication tokens.
*   **Deliverable:** Bulk subsetting of 2010–2023 data for all 7 surface variables and 3D GLORYS target (levels 0–35).
*   **Owner:** Member 1.

## PHASE 2: Production Data Pipeline & Preprocessing
*   **Deliverable:** Automated script to convert raw downloaded NetCDFs into daily preprocessed `.pt` or `.h5` tensors (`[14, 101, 241]` and `[15, 101, 241]`).
*   **Owner:** Member 2.

## PHASE 3: Baselines Training
*   **Deliverable:** Train `PointwiseMLP` and `SimpleCNN` from `src/models/baselines.py`. Compute depth-wise baseline RMSE and MAE.
*   **Owner:** Member 3.

## PHASE 4: OceanEmbed Architecture Optimization
*   **Deliverable:** Train `OceanEmbedNet` on 2010–2019 data. Validate on 2020–2021. Tune latent embedding dimension (64 vs 128) and depth conditioning.
*   **Owner:** Member 4.

## PHASE 5: ARGO Independent Validation
*   **Deliverable:** Ingest 2022–2023 Coriolis/INCOIS ARGO floats via `src/validation/argo.py`. Evaluate test-year predictions against independent in-situ ground truth.
*   **Owner:** Member 5.

## PHASE 6: Streamlit Demo Application & Final Presentation
*   **Deliverable:** Interactive web application for spatial clicking and vertical temperature profile reconstruction against ARGO floats.
*   **Owner:** Member 6.
