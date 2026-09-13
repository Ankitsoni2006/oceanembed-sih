import torch
from torch.utils.data import Dataset
import numpy as np

class OceanDataset(Dataset):
    """
    Standard PyTorch Dataset for SIH26066.
    Accepts aligned input tensors and target tensors.
    """
    def __init__(self, x_data, y_data):
        """
        x_data: [Time, 14, Lat, Lon] (Variables + Masks combined)
        y_data: [Time, 15, Lat, Lon]
        """
        self.x_data = x_data
        self.y_data = y_data
        
        # Verify alignment
        assert self.x_data.shape[0] == self.y_data.shape[0], "Temporal mismatch between inputs and targets"

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        
        # Ensure we output tensors
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).float()
            
        return x, y
