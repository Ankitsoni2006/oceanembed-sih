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


def compute_paired_metrics(
    predictions,
    observations,
    min_samples: int = 3,
    require_variance: bool = True,
):
    """
    Computes RMSE / MAE / Bias / Pearson r over PAIRED 1-D sample vectors.

    Used for observation-vs-model comparison (e.g. independent in-situ ARGO
    profiling floats), where each entry is a single point measurement rather than
    a dense grid. Any pair containing a NaN/None on either side is dropped before
    computation, so a missing observation can never bias a statistic.

    Parameters
    ----------
    predictions, observations : array-like of equal length
        Paired model predictions and observed values (degC).
    min_samples : int
        Minimum number of valid pairs required for RMSE/MAE/Bias to be reported.
    require_variance : bool
        When True, Pearson r is only reported if both series are non-constant.

    Returns
    -------
    dict with keys: count, rmse, mae, bias, corr (None when not computable).
    """
    p = np.asarray(predictions, dtype=np.float64).ravel()
    o = np.asarray(observations, dtype=np.float64).ravel()

    if p.size != o.size:
        raise ValueError(
            f"Paired metric inputs must have equal length, got {p.size} and {o.size}."
        )

    valid = (~np.isnan(p)) & (~np.isnan(o))
    p_v = p[valid]
    o_v = o[valid]
    n = int(p_v.size)

    result = {"count": n, "rmse": None, "mae": None, "bias": None, "corr": None}
    if n < min_samples:
        return result

    diff = p_v - o_v
    result["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    result["mae"] = float(np.mean(np.abs(diff)))
    result["bias"] = float(np.mean(diff))

    if n > 2:
        p_std = float(np.std(p_v))
        o_std = float(np.std(o_v))
        if not require_variance or (p_std > 1e-8 and o_std > 1e-8):
            corr = float(np.corrcoef(p_v, o_v)[0, 1])
            if not np.isnan(corr):
                result["corr"] = corr

    return result

