"""
SIH26066 — Phase 4 Stage 2: Zero-Leakage Temporal Split & Scaler Computation
Strictly fits OceanStandardScaler parameters on Training Partition (Jan 1 – Jul 31, 2020).
Zero leakage into Validation (Aug 2020) or Test (Sep 2020).
"""

import os
import sys
import json
import time
import gc
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.normalization import OceanStandardScaler
from src.data.catalog import SURFACE_CHANNELS

def compute_experiment_scaler(
    processed_dir="data/processed",
    train_chunks=["chunk_2020_01.pt", "chunk_2020_02.pt", "chunk_2020_03.pt", "chunk_2020_04.pt", "chunk_2020_05.pt", "chunk_2020_06.pt", "chunk_2020_07.pt"],
    output_scaler_path="configs/scaler_params_experiment_2020.json"
):
    print("=" * 80)
    print("SIH26066 — PHASE 4 STAGE 2: ZERO-LEAKAGE EXPERIMENT SCALER")
    print("=" * 80)
    print(f"Training Chunks: {train_chunks}")
    print(f"Target Output Path: {output_scaler_path}")

    num_channels = 7
    channel_names = [c[0] for c in SURFACE_CHANNELS]
    total_counts = np.zeros(num_channels, dtype=np.int64)
    sum_vals = np.zeros(num_channels, dtype=np.float64)
    sum_sq_vals = np.zeros(num_channels, dtype=np.float64)
    min_vals = np.full(num_channels, np.inf)
    max_vals = np.full(num_channels, -np.inf)

    all_train_dates = []
    t0 = time.time()

    for cf in train_chunks:
        cpath = os.path.join(processed_dir, cf)
        if not os.path.exists(cpath):
            raise FileNotFoundError(f"Training chunk missing: {cpath}")
        data = torch.load(cpath, weights_only=False)
        X_chunk = data["X"].numpy()
        dates = data["dates"]
        all_train_dates.extend(dates)

        for c in range(num_channels):
            var_data = X_chunk[:, c]
            mask_data = X_chunk[:, c + num_channels]
            valid = (mask_data == 1.0) & (~np.isnan(var_data)) & (~np.isinf(var_data))
            pixels = var_data[valid].astype(np.float64)
            total_counts[c] += len(pixels)
            sum_vals[c] += np.sum(pixels)
            sum_sq_vals[c] += np.sum(pixels ** 2)
            if len(pixels) > 0:
                min_vals[c] = min(min_vals[c], float(np.min(pixels)))
                max_vals[c] = max(max_vals[c], float(np.max(pixels)))

        del data, X_chunk
        gc.collect()

    means = sum_vals / np.maximum(total_counts, 1)
    variances = (sum_sq_vals / np.maximum(total_counts, 1)) - (means ** 2)
    stds = np.sqrt(np.maximum(variances, 1e-8))

    scaler = OceanStandardScaler(num_physical_channels=num_channels)
    scaler.means = means.astype(np.float32)
    scaler.stds = stds.astype(np.float32)
    scaler.is_fitted = True

    os.makedirs(os.path.dirname(output_scaler_path), exist_ok=True)
    scaler.save(output_scaler_path)

    print(f"\nSuccessfully fitted and saved scaler to {output_scaler_path} in {time.time()-t0:.2f}s:")
    print(f"Training dates: {len(all_train_dates)} days ({all_train_dates[0]} to {all_train_dates[-1]})")
    print(f"  {'Channel':<10} | {'Name':<22} | {'Valid Pixels':<15} | {'Mean':<10} | {'Std':<10} | {'Min':<8} | {'Max':<8}")
    print("  " + "-" * 90)
    for c in range(num_channels):
        print(f"  Ch {c:<7} | {channel_names[c]:<22} | {total_counts[c]:<15,d} | {means[c]:<10.4f} | {stds[c]:<10.4f} | {min_vals[c]:<8.2f} | {max_vals[c]:<8.2f}")

    # Metadata file for reproducibility & audit
    meta = {
        "scaler_file": output_scaler_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_period": {
            "start": all_train_dates[0],
            "end": all_train_dates[-1],
            "num_days": len(all_train_dates),
            "chunks": train_chunks
        },
        "validation_period": {
            "start": "2020-08-01",
            "end": "2020-08-31",
            "num_days": 31,
            "chunks": ["chunk_2020_08.pt"]
        },
        "test_period": {
            "start": "2020-09-01",
            "end": "2020-09-30",
            "num_days": 30,
            "chunks": ["chunk_2020_09.pt"]
        },
        "channels": {
            channel_names[c]: {
                "channel_idx": c,
                "mean": float(means[c]),
                "std": float(stds[c]),
                "valid_pixel_count": int(total_counts[c]),
                "min": float(min_vals[c]),
                "max": float(max_vals[c])
            } for c in range(num_channels)
        }
    }
    meta_path = output_scaler_path.replace(".json", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved split metadata to: {meta_path}\n")

if __name__ == "__main__":
    compute_experiment_scaler()
