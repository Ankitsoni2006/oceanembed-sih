import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    """
    Primary Loss function. 
    Applies MSE only where target values are valid (not NaN, inside ocean).
    Safely zeroes out NaN elements before computing squared error to prevent
    IEEE 754 NaN * 0.0 = NaN leakage into gradients.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, valid_mask=None):
        target_valid = ~torch.isnan(target)
        
        if valid_mask is not None:
            combined_mask = target_valid & valid_mask
        else:
            combined_mask = target_valid
            
        mask_f = combined_mask.float()
        
        # Replace NaNs in target with 0 before subtracting
        safe_target = torch.where(combined_mask, target, torch.zeros_like(target))
        
        diff = (pred - safe_target) * mask_f
        loss = diff ** 2
        
        total_valid = mask_f.sum()
        if total_valid < 1.0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        return loss.sum() / total_valid
