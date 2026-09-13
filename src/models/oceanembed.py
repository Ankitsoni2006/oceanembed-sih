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

class OceanEmbedNet(nn.Module):
    """
    Masked Multi-Scale U-Net Encoder -> Latent Ocean Embedding -> Depth-Conditioned Decoder
    """
    def __init__(self, in_vars=7, num_depths=15, base_features=32, embedding_dim=128):
        super().__init__()
        # 7 variables + 7 masks = 14 input channels
        self.in_channels = in_vars * 2
        self.num_depths = num_depths
        self.embedding_dim = embedding_dim
        
        # Multi-scale U-Net Encoder
        self.enc1 = DoubleConv(self.in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_features, base_features*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_features*2, base_features*4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Latent Ocean Embedding Bottleneck
        self.bottleneck = DoubleConv(base_features*4, embedding_dim)
        
        # Depth representation
        self.depth_embedding = nn.Embedding(num_depths, embedding_dim)
        
        # Depth-conditioned decoder (shared across depths)
        self.up3 = nn.ConvTranspose2d(embedding_dim * 2, base_features*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features*8, base_features*4)
        
        self.up2 = nn.ConvTranspose2d(base_features*4, base_features*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features*4, base_features*2)
        
        self.up1 = nn.ConvTranspose2d(base_features*2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features*2, base_features)
        self.final_conv = nn.Conv2d(base_features, 1, kernel_size=1)
        
        # Physical climatology temperature prior across the 15 standard depths (0m down to 1000m)
        # Allows the deep network to predict temperature anomalies relative to oceanic stratification
        climatology = torch.tensor([28.2, 28.1, 27.9, 27.5, 26.8, 25.0, 22.5, 19.8, 17.2, 15.1, 13.0, 10.5, 8.4, 7.0, 5.2], dtype=torch.float32)
        self.climatology_prior = nn.Parameter(climatology.view(num_depths, 1, 1, 1))

    def forward(self, x, depth_indices):
        """
        x: [B, 14, H, W] tensor (variables and masks combined)
        depth_indices: list or tensor of indices representing target depths
        """
        B = x.size(0)
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Latent Ocean Embedding
        latent = self.bottleneck(self.pool3(e3)) # [B, embedding_dim, H/8, W/8]
        
        outputs = []
        # Decode for each depth independently using shared weights but conditioned embedding
        for d in depth_indices:
            # Condition the latent embedding with the target depth
            d_emb = self.depth_embedding(d) # [embedding_dim]
            # Correctly expand to match [B, embedding_dim, H_latent, W_latent] without double-multiplying B
            d_emb = d_emb.view(1, self.embedding_dim, 1, 1).expand_as(latent)
            
            combined = torch.cat([latent, d_emb], dim=1) # [B, 2*embedding_dim, H/8, W/8]
            
            # Decoder
            d3 = self.up3(combined)
            diffY = e3.size()[2] - d3.size()[2]
            diffX = e3.size()[3] - d3.size()[3]
            d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            d3 = self.dec3(torch.cat([e3, d3], dim=1))
            
            d2 = self.up2(d3)
            diffY = e2.size()[2] - d2.size()[2]
            diffX = e2.size()[3] - d2.size()[3]
            d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            d2 = self.dec2(torch.cat([e2, d2], dim=1))
            
            d1 = self.up1(d2)
            diffY = e1.size()[2] - d1.size()[2]
            diffX = e1.size()[3] - d1.size()[3]
            d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            d1 = self.dec1(torch.cat([e1, d1], dim=1))
            
            out = self.final_conv(d1) + self.climatology_prior[d] # [B, 1, H, W]
            outputs.append(out)
            
        return torch.cat(outputs, dim=1) # [B, num_depths, H, W]
