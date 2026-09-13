import numpy as np
import torch

def generate_valid_mask(tensor, missing_val_threshold=-999.0):
    """
    Generates a boolean/float validity mask for any input tensor or array.
    Distinguishes an actual value near zero from a missing value encoded as zero/NaN.
    """
    if isinstance(tensor, torch.Tensor):
        mask = (~torch.isnan(tensor)) & (tensor > missing_val_threshold)
        return mask.float()
    else:
        # Numpy fallback
        mask = (~np.isnan(tensor)) & (tensor > missing_val_threshold)
        return mask.astype(np.float32)

def concatenate_variables_and_masks(variables_tensor, missing_val_threshold=-999.0):
    """
    Creates the 14-channel input by appending masks to variables.
    variables_tensor: [B, 7, H, W]
    Returns: [B, 14, H, W]
    """
    masks = generate_valid_mask(variables_tensor, missing_val_threshold)
    # Replace NaNs in variables with 0 so the convolutions don't spread NaNs
    # The network will learn to ignore these areas based on the mask
    safe_vars = torch.where(torch.isnan(variables_tensor), torch.zeros_like(variables_tensor), variables_tensor)
    return torch.cat([safe_vars, masks], dim=1)
