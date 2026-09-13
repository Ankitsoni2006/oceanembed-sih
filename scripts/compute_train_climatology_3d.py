"""
SIH26066 — Zero-Leakage 3D Training Climatology Prior
Computes the mean temperature field across the 213 training days (Jan 1 - Jul 31, 2020)
strictly from chunk_2020_01.pt through chunk_2020_07.pt.
Zero leakage into August validation or September test.
Saves to data/processed/train_climatology_3d.pt.
"""

import os
import sys
import gc
import torch
import numpy as np

def compute_train_climatology():
    print("=" * 80)
    print("SIH26066 — COMPUTING ZERO-LEAKAGE 3D TRAINING CLIMATOLOGY")
    print("=" * 80)

    train_chunks = [f"chunk_2020_{m:02d}.pt" for m in range(1, 8)]
    num_depths = 15
    H, W = 101, 241

    sum_t = torch.zeros(num_depths, H, W, dtype=torch.float64)
    count_t = torch.zeros(num_depths, H, W, dtype=torch.int64)

    total_days = 0
    for cf in train_chunks:
        cpath = os.path.join("data/processed", cf)
        data = torch.load(cpath, weights_only=False)
        Y_chunk = data["Y"] # [N, 15, 101, 241]
        n_days = Y_chunk.shape[0]
        total_days += n_days

        for d in range(num_depths):
            y_d = Y_chunk[:, d] # [N, H, W]
            valid = ~torch.isnan(y_d)
            # Replace NaNs with 0 for summing
            y_d_clean = torch.where(valid, y_d.double(), torch.tensor(0.0, dtype=torch.float64))
            sum_t[d] += y_d_clean.sum(dim=0)
            count_t[d] += valid.long().sum(dim=0)

        del data, Y_chunk
        gc.collect()

    mean_t = torch.where(count_t > 0, (sum_t / torch.clamp(count_t, min=1)).float(), torch.tensor(float('nan'), dtype=torch.float32))

    # Fill land/unobserved cells with depth-wise scalar mean so network doesn't receive NaNs in prior
    for d in range(num_depths):
        slice_d = mean_t[d]
        valid_px = ~torch.isnan(slice_d)
        if valid_px.any():
            d_mean = slice_d[valid_px].mean().item()
            mean_t[d] = torch.where(valid_px, slice_d, torch.tensor(d_mean, dtype=torch.float32))

    print(f"Computed 3D training climatology across {total_days} days.")
    for d in range(num_depths):
        print(f"  Depth idx {d:2d}: Mean = {mean_t[d].mean():.2f} C | Min = {mean_t[d].min():.2f} C | Max = {mean_t[d].max():.2f} C")

    out_path = "data/processed/train_climatology_3d.pt"
    torch.save({"train_climatology_3d": mean_t, "num_days": total_days, "train_chunks": train_chunks}, out_path)
    print(f"\nSaved zero-leakage 3D prior to {out_path}")

if __name__ == "__main__":
    compute_train_climatology()
