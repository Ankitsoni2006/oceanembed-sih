"""
SIH26066 — OceanEmbedNetV3_Decoder
Architecture Enhancement for Phase 5B (Experiment 2).
Replaces the sequential 1-channel shared decoder loop with a unified Multi-Scale U-Net
and a dedicated multi-depth projection head, directly predicting all 15 depths simultaneously.
Preserves the 14-channel input, Multi-Scale Encoder, and 128-dim Latent Ocean Embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class OceanEmbedNetV3_Decoder(nn.Module):
    """
    Unified Multi-Scale U-Net Encoder -> 128-dim Latent Ocean Embedding -> Multi-Depth Spatial Decoder.
    Directly predicts all 15 depths simultaneously, allocating dedicated vertical projection capacity.
    """
    def __init__(self, in_vars=7, num_depths=15, base_features=32, embedding_dim=128):
        super().__init__()
        self.in_channels = in_vars * 2 # 14 channels
        self.num_depths = num_depths
        self.embedding_dim = embedding_dim

        # Multi-scale U-Net Encoder
        self.enc1 = DoubleConv(self.in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Latent Ocean Embedding Bottleneck
        self.bottleneck = DoubleConv(base_features * 4, embedding_dim)

        # Spatial Decoder with Multi-Scale Skip Connections
        self.up3 = nn.ConvTranspose2d(embedding_dim, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features * 2, base_features)

        # Dedicated Multi-Depth Projection Head (Maps full-resolution features directly to 15 depths)
        self.final_conv = nn.Conv2d(base_features, num_depths, kernel_size=1)

        # Physical ocean climatological stratification prior
        climatology = torch.tensor([28.2, 28.1, 27.9, 27.5, 26.8, 25.0, 22.5, 19.8, 17.2, 15.1, 13.0, 10.5, 8.4, 7.0, 5.2], dtype=torch.float32)
        self.climatology_prior = nn.Parameter(climatology.view(1, num_depths, 1, 1))

        # Initialize final conv bias to zero so climatology prior serves as exact initial baseline
        with torch.no_grad():
            self.final_conv.bias.zero_()

    def forward(self, x, depth_indices=None):
        """
        x: [B, 14, 101, 241]
        depth_indices: Optional list of depth indices. If None, returns all 15 depths.
        """
        # Encoder
        e1 = self.enc1(x)                                 # [B, 32, 101, 241]
        e2 = self.enc2(self.pool1(e1))                    # [B, 64, 50, 120]
        e3 = self.enc3(self.pool2(e2))                    # [B, 128, 25, 60]

        # Latent Ocean Embedding
        latent = self.bottleneck(self.pool3(e3))          # [B, 128, 12, 30]

        # Decoder with Skip Connections
        d3 = self.up3(latent)                             # [B, 128, 24, 60]
        diffY = e3.size(2) - d3.size(2)
        diffX = e3.size(3) - d3.size(3)
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = self.dec3(torch.cat([e3, d3], dim=1))       # [B, 128, 25, 60]

        d2 = self.up2(d3)                                 # [B, 64, 50, 120]
        diffY = e2.size(2) - d2.size(2)
        diffX = e2.size(3) - d2.size(3)
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = self.dec2(torch.cat([e2, d2], dim=1))       # [B, 64, 50, 120]

        d1 = self.up1(d2)                                 # [B, 32, 100, 240]
        diffY = e1.size(2) - d1.size(2)
        diffX = e1.size(3) - d1.size(3)
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = self.dec1(torch.cat([e1, d1], dim=1))       # [B, 32, 101, 241]

        # Multi-Depth Output
        out = self.final_conv(d1) + self.climatology_prior # [B, 15, 101, 241]

        if depth_indices is not None:
            out = out[:, depth_indices]

        return out
