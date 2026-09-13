"""
SIH26066 — Phase 4 Stage 6: OceanEmbedNet Training & Out-of-Sample Evaluation
Trains the Masked Multi-Scale U-Net Encoder -> Latent Ocean Embedding -> Depth-Conditioned Decoder
on Jan–Jul 2020 (213 days), validates on Aug 2020 (31 days), and tests on Sep 2020 (30 days).
Uses identical zero-leakage scaler, splits, and loss function as baselines for strict scientific fairness.
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
from src.models.oceanembed import OceanEmbedNet
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


def evaluate_model_full(model, dataset: MultiChunkDataset, depth_values: List[float], batch_size: int = 4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)

    all_preds = []
    all_targets = []
    all_dates = []

    with torch.no_grad():
        for x_b, y_b, dt_b in loader:
            pred = model(x_b, depth_indices)
            all_preds.append(pred)
            all_targets.append(y_b)
            all_dates.extend(dt_b)

    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)

    overall = calculate_metrics(cat_preds, cat_targets)
    depth_metrics = evaluate_depth_wise(cat_preds, cat_targets, depth_values)

    return {
        "overall": overall,
        "depth_wise": {f"{int(d)}m": m for d, m in depth_metrics.items()},
        "num_samples": len(all_dates),
        "date_start": all_dates[0],
        "date_end": all_dates[-1]
    }


def main():
    parser = argparse.ArgumentParser(description="Train OceanEmbedNet on 2020 Partition")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default 5)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default 4)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("SIH26066 — PHASE 4 STAGE 6: OCEANEMBEDNET TRAINING")
    print("=" * 80)
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr} | Seed: {args.seed}")

    processed_dir = "data/processed"
    scaler_path = "configs/scaler_params_experiment_2020.json"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports/results", exist_ok=True)

    # 1. Dataset splits
    train_chunks = [f"chunk_2020_{m:02d}.pt" for m in range(1, 8)]  # Jan–Jul (213 days)
    val_chunks   = ["chunk_2020_08.pt"]                              # Aug (31 days)
    test_chunks  = ["chunk_2020_09.pt"]                              # Sep (30 days)

    print(f"Loading datasets with zero-leakage scaler from {scaler_path}...")
    scaler = OceanStandardScaler.load(scaler_path)

    t0 = time.time()
    train_ds = MultiChunkDataset(processed_dir, train_chunks, scaler=scaler)
    val_ds   = MultiChunkDataset(processed_dir, val_chunks, scaler=scaler)
    test_ds  = MultiChunkDataset(processed_dir, test_chunks, scaler=scaler)
    print(f"Loaded datasets in {time.time()-t0:.2f}s:")
    print(f"  Train: {len(train_ds)} samples ({train_ds.index[0][2]} to {train_ds.index[-1][2]})")
    print(f"  Val:   {len(val_ds)} samples ({val_ds.index[0][2]} to {val_ds.index[-1][2]})")
    print(f"  Test:  {len(test_ds)} samples ({test_ds.index[0][2]} to {test_ds.index[-1][2]})")

    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)
    loss_fn = MaskedMSELoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Instantiate OceanEmbedNet (Locked Architecture)
    model = OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture: OceanEmbedNet (U-Net Encoder -> Latent Embedding -> Conditioned Decoder)")
    print(f"Total Trainable Parameters: {param_count:,}")

    ckpt_path = "checkpoints/oceanembed_best.pt"
    latest_path = "checkpoints/oceanembed_latest.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = []
    t_train_start = time.time()

    print("\n" + "=" * 70)
    print(f"STARTING TRAINING — {args.epochs} EPOCHS (54 batches per epoch)")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for b_idx, (x_b, y_b, _) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(x_b, depth_indices)
            loss = loss_fn(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch:02d}/{args.epochs:02d}] Batch {b_idx+1:02d}/{len(train_loader)} | Running Batch Loss: {loss.item():.4f}")

        train_loss /= max(1, n_batches)

        # Validation pass
        model.eval()
        val_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for vx, vy, _ in val_loader:
                vpred = model(vx, depth_indices)
                vloss = loss_fn(vpred, vy)
                val_loss += vloss.item()
                v_batches += 1
        val_loss /= max(1, v_batches)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        ep_time = time.time() - ep_t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss, "val_rmse": float(np.sqrt(val_loss))}, ckpt_path)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}, latest_path)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_rmse": round(float(np.sqrt(val_loss)), 4),
            "lr": round(current_lr, 6),
            "epoch_time_s": round(ep_time, 2),
            "is_best": is_best
        })

        print(f"\n>>> Epoch {epoch:02d}/{args.epochs:02d} Finished in {ep_time:.1f}s ({ep_time/60:.2f} min) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (RMSE: {np.sqrt(val_loss):.4f}°C) {'*** BEST CHECKPOINT' if is_best else ''}\n")

    total_train_time = time.time() - t_train_start
    print(f"\nAll {args.epochs} epochs completed in {total_train_time:.1f}s ({total_train_time/60:.2f} min).")
    print(f"Best Validation Loss: {best_val_loss:.4f} (RMSE: {np.sqrt(best_val_loss):.4f}°C)")

    # Save history
    with open("reports/results/oceanembed_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 3. Final Out-of-Sample Evaluation using best checkpoint
    print("\n" + "=" * 70)
    print("FINAL SCIENTIFIC EVALUATION ON HELD-OUT SEPTEMBER 2020 TEST PERIOD")
    print("=" * 70)
    best_ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from Epoch {best_ckpt['epoch']} (Val Loss: {best_ckpt['val_loss']:.4f})")

    # August Validation
    val_eval = evaluate_model_full(model, val_ds, depth_values, batch_size=args.batch_size)
    print(f"\n  [August 2020 Validation Set]")
    print(f"    RMSE: {val_eval['overall']['rmse']:.4f}°C | MAE: {val_eval['overall']['mae']:.4f}°C | Bias: {val_eval['overall']['bias']:+.4f}°C | Corr: {val_eval['overall']['corr']:.4f}")

    # September Test
    test_eval = evaluate_model_full(model, test_ds, depth_values, batch_size=args.batch_size)
    print(f"\n  [September 2020 Held-Out Test Set]")
    print(f"    RMSE: {test_eval['overall']['rmse']:.4f}°C | MAE: {test_eval['overall']['mae']:.4f}°C | Bias: {test_eval['overall']['bias']:+.4f}°C | Corr: {test_eval['overall']['corr']:.4f}")

    # Print Depth-wise table
    print("\n" + "-" * 75)
    print(f" {'Depth':>8s} | {'Test RMSE (°C)':>15s} | {'Test MAE (°C)':>15s} | {'Test Bias (°C)':>15s} | {'Pearson r':>10s} ")
    print("-" * 75)
    for d in depth_values:
        d_key = f"{int(d)}m"
        dm = test_eval["depth_wise"][d_key]
        print(f" {d_key:>8s} | {dm['rmse']:>15.4f} | {dm['mae']:>15.4f} | {dm['bias']:>+15.4f} | {dm['corr']:>10.4f} ")
    print("-" * 75)

    oceanembed_results = {
        "model_name": "OceanEmbedNet",
        "display_name": "OceanEmbedNet (Masked Multi-Scale U-Net + Latent Embedding)",
        "parameters": param_count,
        "training_period": {"start": train_ds.index[0][2], "end": train_ds.index[-1][2], "num_days": len(train_ds)},
        "val_period":   {"start": val_ds.index[0][2], "end": val_ds.index[-1][2], "num_days": len(val_ds)},
        "test_period":  {"start": test_ds.index[0][2], "end": test_ds.index[-1][2], "num_days": len(test_ds)},
        "scaler_used": scaler_path,
        "training_duration_s": round(total_train_time, 2),
        "best_epoch": best_ckpt["epoch"],
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": ckpt_path,
        "val_metrics": val_eval,
        "test_metrics": test_eval,
        "history": history
    }

    out_json = "reports/results/oceanembed_results.json"
    with open(out_json, "w") as f:
        json.dump(oceanembed_results, f, indent=2)
    print(f"\nSaved OceanEmbedNet results to: {out_json}")


if __name__ == "__main__":
    main()
