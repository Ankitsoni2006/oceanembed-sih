import torch
import torch.nn as nn

class PointwiseMLP(nn.Module):
    """
    Baseline 3: Point-wise MLP regression.
    Treats every spatial pixel independently, ignoring surrounding context.
    """
    def __init__(self, in_vars=7, num_depths=15, hidden_dim=64):
        super().__init__()
        # Input is 7 vars + 7 masks = 14
        self.net = nn.Sequential(
            nn.Linear(in_vars * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_depths)
        )
        # Initialize final layer bias to physical ocean climatological profile
        climatology = torch.tensor([28.2, 28.1, 27.9, 27.5, 26.8, 25.0, 22.5, 19.8, 17.2, 15.1, 13.0, 10.5, 8.4, 7.0, 5.2], dtype=torch.float32)
        with torch.no_grad():
            self.net[-1].bias.copy_(climatology)
        
    def forward(self, x, depth_indices=None):
        B, C, H, W = x.shape
        # Permute to put channels last, then flatten spatial dims
        x = x.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, C]
        out = self.net(x) # [B*H*W, num_depths]
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2) # [B, num_depths, H, W]
        return out

class SimpleCNNBaseline(nn.Module):
    """
    Baseline 4: Simple spatial CNN without U-Net skip connections or latent bottleneck.
    """
    def __init__(self, in_vars=7, num_depths=15, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_vars * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_depths, 1)
        )
        # Initialize final conv bias to physical ocean climatological profile
        climatology = torch.tensor([28.2, 28.1, 27.9, 27.5, 26.8, 25.0, 22.5, 19.8, 17.2, 15.1, 13.0, 10.5, 8.4, 7.0, 5.2], dtype=torch.float32)
        with torch.no_grad():
            self.net[-1].bias.copy_(climatology)
        
    def forward(self, x, depth_indices=None):
        return self.net(x)
