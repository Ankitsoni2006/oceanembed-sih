import torch
import numpy as np

def calculate_metrics(pred, target, valid_mask=None):
    """
    Calculates evaluation metrics over valid pixels only.
    """
    target_valid = ~torch.isnan(target)
    
    if valid_mask is not None:
        mask = target_valid & valid_mask
    else:
        mask = target_valid
        
    pred_flat = pred[mask]
    target_flat = target[mask]
    
    if len(pred_flat) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'corr': np.nan}
        
    rmse = torch.sqrt(torch.mean((pred_flat - target_flat)**2)).item()
    mae = torch.mean(torch.abs(pred_flat - target_flat)).item()
    bias = torch.mean(pred_flat - target_flat).item()
    
    # Pearson correlation
    pred_mean = torch.mean(pred_flat)
    target_mean = torch.mean(target_flat)
    cov = torch.mean((pred_flat - pred_mean) * (target_flat - target_mean))
    pred_std = torch.std(pred_flat, unbiased=False)
    target_std = torch.std(target_flat, unbiased=False)
    corr = (cov / (pred_std * target_std + 1e-8)).item()
    
    return {'rmse': rmse, 'mae': mae, 'bias': bias, 'corr': corr}

def evaluate_depth_wise(pred, target, depths):
    """
    Calculates metrics for each depth channel independently.
    pred, target: [B, num_depths, H, W]
    """
    metrics = {}
    for i, d in enumerate(depths):
        d_pred = pred[:, i, :, :]
        d_target = target[:, i, :, :]
        metrics[d] = calculate_metrics(d_pred, d_target)
    return metrics
