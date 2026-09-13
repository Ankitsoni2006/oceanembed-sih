# OceanEmbed Model Architecture

This document describes the proposed model for SIH26066.

## Inputs
**14-Channel Tensor:**
- 7 Surface Variables (SST, SSS, SSH, U, V, Wind U, Wind V)
- 7 Binary Validity Masks (1 for valid ocean, 0 for missing/land)

## Core Pipeline (`OceanEmbedNet`)
1. **Masked Multi-Scale U-Net Encoder:** Uses standard convolution blocks with max pooling to extract spatial context and features from the 14-channel input.
2. **Latent Ocean Embedding:** A spatial bottleneck layer capturing the joint representation of surface dynamics.
3. **Depth-Conditioned Profile Decoder:** Instead of 15 independent output regression heads, the decoder shares weights across all depths. It injects a learned depth embedding into the latent space, forcing the decoder to conceptually "navigate" down the water column, outputting one temperature map at a time, resulting in physically coherent vertical profiles.

## Baselines
We evaluate the core pipeline against:
- Climatological means
- Point-wise MLP (ignores spatial context)
- Simple CNN (ignores multi-scale bottleneck context)
