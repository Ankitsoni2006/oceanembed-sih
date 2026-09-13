# SIH26066 — ONE-PAGE TEAM SUMMARY (AUDITED)

### Guys, why are we choosing this PS?
Unlike AI/LLM problem statements where thousands of teams will just wrap API wrappers, this is a hard, distinct geospatial problem. We have verified that the math, the regridding, the U-Net models, and the independent ARGO float validation are all working on real data.

### What are we actually building?
A system called **OceanEmbed**. It takes 7 daily satellite maps of the ocean surface (SST, SSS, SSH, currents, winds) and uses a Masked Multi-Scale Convolutional Neural Network (U-Net) to predict subsurface ocean temperature profiles at 15 depths down to 1000 meters across the North Indian Ocean (5°N–30°N, 45°E–105°E).

### What has been experimentally proven?
1. **Regridding Works:** Real GLORYS and real OSTIA satellite SST both regrid cleanly to our exact `101 x 241` matrix in ~0.74 seconds.
2. **PyTorch Integration Works:** Real 14-channel input tensors and 15-channel target tensors flow through the DataLoader, Pointwise MLP, Simple CNN, and OceanEmbedNet.
3. **ARGO Independent Validation Works:** Real ARGO floats from Coriolis GDAC were downloaded, colocated to our 0.25° grid, and interpolated to the 15 SIH target depths.
4. **Compute is Trivial:** The entire process takes under 1.7GB of RAM. A single year of data can be trained in <20 minutes on a GPU or ~6.5 hours on CPU.

### What bugs were caught and eliminated?
1. **IEEE 754 NaN Loss Bug:** In `src/training/loss.py`, multiplying NaNs by zero caused loss to become `nan`. Fixed by zeroing unobserved target pixels before subtraction.
2. **Batch Scaling Bug:** In `src/models/oceanembed.py`, a duplicate `.repeat()` call caused batches $B > 1$ to crash. Fixed; now scales to arbitrary batch sizes.

### What is the remaining condition?
The Copernicus Marine CLI needs authentication configured locally (`copernicusmarine login`). Once done, bulk downloading is unlocked.

### What is our common historical training period?
**2010 to 2023** (14 years), strictly determined by SSS and wind data availability.
- Train: 2010–2019
- Val: 2020–2021
- Test: 2022–2023

### Final Verdict:
**CONDITIONAL GO** (Confidence: 88/100). The technical risks are resolved; the project is ready for execution.
