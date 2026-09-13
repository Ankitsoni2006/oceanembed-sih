"""
SIH26066 — Phase 5B: Experiment 2 (Improved Multi-Depth Spatial Decoder)
Trains OceanEmbedNetV3_Decoder on Jan-Jul 2020 (213 days).
Evaluates on August 2020 validation set (31 days).
Saves checkpoint to checkpoints/phase5/oceanembed_v3_decoder.pt.
"""

import os
import sys
import json
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
from src.models.oceanembed_v3_decoder import OceanEmbedNetV3_Decoder
from src.preprocessing.normalization import OceanStandardScaler
from src.training.loss import MaskedMSELoss
from src.data.catalog import TARGET_DEPTHS
from src.evaluation.metrics import calculate_metrics, evaluate_depth_wise

class MultiChunkDataset(Dataset):
    def __init__(self, processed_dir: str, chunk_files: List[str], scaler: OceanStandardScaler = None):
        self.processed_dir = processed_dir
        self.scaler = scaler
        self.chunk_files = chunk_files
        self.index = []
        self._chunks = {}

        for cf in chunk_files:
            cpath = os.path.join(processed_dir, cf)
            data = torch.load(cpath, weights_only=False)
            self._chunks[cf] = (data["X"], data["Y"])
            dates = data["dates"]
            for offset in range(len(dates)):
                self.index.append((cf, offset, dates[offset]))
            del data
            gc.collect()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        cf, offset, dt_str = self.index[idx]
        x_chunk, y_chunk = self._chunks[cf]
        x = x_chunk[offset]
        y = y_chunk[offset]
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x, y, dt_str

def run_experiment_2():
    print("=" * 80)
    print("SIH26066 — PHASE 5B: EXPERIMENT 2 (IMPROVED MULTI-DEPTH DECODER)")
    print("=" * 80)

    os.makedirs("checkpoints/phase5", exist_ok=True)
    os.makedirs("reports/phase5", exist_ok=True)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    processed_dir = "data/processed"
    scaler_path = "configs/scaler_params_experiment_2020.json"
    scaler = OceanStandardScaler.load(scaler_path)

    train_chunks = [f"chunk_2020_{m:02d}.pt" for m in range(1, 8)]  # Jan–Jul (213 days)
    val_chunks   = ["chunk_2020_08.pt"]                              # Aug (31 days)

    print(f"Loading Training Chunks (Jan-Jul: {len(train_chunks)} chunks)...")
    train_ds = MultiChunkDataset(processed_dir, train_chunks, scaler=scaler)
    val_ds   = MultiChunkDataset(processed_dir, val_chunks, scaler=scaler)
    print(f"  Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

    # Initialize OceanEmbedNetV3_Decoder
    model = OceanEmbedNetV3_Decoder(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture: OceanEmbedNetV3_Decoder")
    print(f"Total Trainable Parameters: {param_count:,}")

    loss_fn = MaskedMSELoss()

    epochs = 12
    lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_rmse = float("inf")
    ckpt_path = "checkpoints/phase5/oceanembed_v3_decoder.pt"
    t_start = time.time()

    print(f"\nTraining OceanEmbedNetV3_Decoder for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x_b, y_b, _ in train_loader:
            optimizer.zero_grad()
            pred = model(x_b) # [B, 15, H, W] directly!
            loss = loss_fn(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for vx, vy, _ in val_loader:
                vpred = model(vx)
                vloss = loss_fn(vpred, vy)
                val_loss += vloss.item()
                val_batches += 1

        val_loss /= max(1, val_batches)
        val_rmse = float(np.sqrt(val_loss))
        scheduler.step()
        ep_elapsed = time.time() - ep_t0

        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "experiment_id": "EXP-02",
                "model_type": "OceanEmbedNetV3_Decoder"
            }, ckpt_path)

        print(f"  Epoch {epoch:02d}/{epochs:02d} [{ep_elapsed:.1f}s] | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} C {'*BEST*' if is_best else ''}")

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.1f}s. Best August Validation RMSE: {best_val_rmse:.4f} C")

    # Evaluate best checkpoint depth-wise on August validation
    best_state = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for vx, vy, _ in val_loader:
            all_preds.append(model(vx))
            all_targets.append(vy)
    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)

    val_metrics = calculate_metrics(cat_preds, cat_targets)
    val_depth_metrics = evaluate_depth_wise(cat_preds, cat_targets, depth_values)

    print("\n--- August 2020 Validation Depth-Wise Performance (EXP-02: Multi-Depth Decoder) ---")
    print(f"Overall August Val: RMSE={val_metrics['rmse']:.4f} C | MAE={val_metrics['mae']:.4f} C | r={val_metrics['corr']:.4f}")
    for d in [50, 75, 100, 125, 150, 200, 500]:
        print(f"  Depth {d:4d}m: RMSE = {val_depth_metrics[float(d)]['rmse']:.4f} C")

    # Log to registry
    reg_entry = {
        "experiment_id": "EXP-02",
        "name": "OceanEmbedNetV3_Decoder (Multi-Depth Spatial Head)",
        "checkpoint": ckpt_path,
        "parameter_count": param_count,
        "training_time_seconds": round(total_time, 1),
        "val_august_rmse": round(val_metrics['rmse'], 4),
        "val_august_mae": round(val_metrics['mae'], 4),
        "val_august_corr": round(val_metrics['corr'], 4),
        "per_depth_rmse": {f"{int(d)}m": round(val_depth_metrics[d]['rmse'], 4) for d in depth_values},
        "comparison_vs_baseline_oe": {
            "baseline_oe_val_rmse": 1.6736,
            "delta_rmse": round(val_metrics['rmse'] - 1.6736, 4)
        },
        "status": "COMPLETED",
        "decision": "EVALUATE CANDIDATE FOR PHASE 5 SELECTION"
    }

    reg_file = "reports/phase5/experiment_registry.json"
    registry = []
    if os.path.exists(reg_file):
        with open(reg_file, "r") as f:
            registry = json.load(f)
    registry = [r for r in registry if r.get("experiment_id") != "EXP-02"]
    registry.append(reg_entry)
    with open(reg_file, "w") as f:
        json.dump(registry, f, indent=2)

    print("\nExperiment 2 recorded in reports/phase5/experiment_registry.json")

if __name__ == "__main__":
    run_experiment_2()
