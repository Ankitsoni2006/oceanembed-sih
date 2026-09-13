"""
SIH26066 — Phase 4 Stage 5: Rigorous Baseline Training & Evaluation Pipeline
Trains PointwiseMLP and SimpleCNNBaseline on Jan–Jul 2020 (213 days),
validates on Aug 2020 (31 days), tests out-of-sample on Sep 2020 (30 days),
and benchmarks against static Climatology baseline.
"""

import os
import sys
import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.preprocessing.normalization import OceanStandardScaler
from src.training.loss import MaskedMSELoss
from src.data.catalog import TARGET_DEPTHS
from src.evaluation.metrics import calculate_metrics, evaluate_depth_wise


class MultiChunkDataset(Dataset):
    """
    Cached multi-chunk dataset. Loads chunk files into RAM cache
    to achieve maximum training speed and zero disk thrashing.
    """
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


def evaluate_climatology(dataset: MultiChunkDataset, depth_values: List[float], batch_size: int = 4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    climatology_profile = torch.tensor([28.2, 28.1, 27.9, 27.5, 26.8, 25.0, 22.5, 19.8, 17.2, 15.1, 13.0, 10.5, 8.4, 7.0, 5.2], dtype=torch.float32)
    
    all_preds = []
    all_targets = []
    all_dates = []

    for _, y_b, dt_b in loader:
        B, D, H, W = y_b.shape
        # Broadcast climatology profile across batch and spatial grid
        pred = climatology_profile.view(1, D, 1, 1).expand(B, D, H, W)
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
    print("=" * 80)
    print("SIH26066 — PHASE 4 STAGE 5: BASELINE MODELS TRAINING & BENCHMARKING")
    print("=" * 80)

    processed_dir = "data/processed"
    scaler_path = "configs/scaler_params_experiment_2020.json"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports/results", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

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
    print(f"Loaded all datasets in {time.time()-t0:.2f}s:")
    print(f"  Train: {len(train_ds)} samples ({train_ds.index[0][2]} to {train_ds.index[-1][2]})")
    print(f"  Val:   {len(val_ds)} samples ({val_ds.index[0][2]} to {val_ds.index[-1][2]})")
    print(f"  Test:  {len(test_ds)} samples ({test_ds.index[0][2]} to {test_ds.index[-1][2]})")

    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)
    loss_fn = MaskedMSELoss()

    batch_size = 4
    epochs = 15
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    results = {
        "benchmark_name": "Phase 4 Stage 5 Baseline Benchmarks",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_period": {"start": train_ds.index[0][2], "end": train_ds.index[-1][2], "num_days": len(train_ds)},
        "val_period":   {"start": val_ds.index[0][2], "end": val_ds.index[-1][2], "num_days": len(val_ds)},
        "test_period":  {"start": test_ds.index[0][2], "end": test_ds.index[-1][2], "num_days": len(test_ds)},
        "scaler_used": scaler_path,
        "models": {}
    }

    # 1. Climatology Baseline
    print("\n" + "=" * 70)
    print("EVALUATING CLIMATOLOGY BASELINE (Static Regional Profile)")
    print("=" * 70)
    clim_val = evaluate_climatology(val_ds, depth_values, batch_size=batch_size)
    clim_test = evaluate_climatology(test_ds, depth_values, batch_size=batch_size)
    print(f"  Val RMSE: {clim_val['overall']['rmse']:.4f}°C | MAE: {clim_val['overall']['mae']:.4f}°C | Bias: {clim_val['overall']['bias']:+.4f}°C | Corr: {clim_val['overall']['corr']:.4f}")
    print(f"  Test RMSE: {clim_test['overall']['rmse']:.4f}°C | MAE: {clim_test['overall']['mae']:.4f}°C | Bias: {clim_test['overall']['bias']:+.4f}°C | Corr: {clim_test['overall']['corr']:.4f}")

    results["models"]["climatology"] = {
        "display_name": "Static Climatology Profile",
        "parameters": 0,
        "val_metrics": clim_val,
        "test_metrics": clim_test
    }

    # 2. Trainable Baselines: PointwiseMLP and SimpleCNNBaseline
    trainable_baselines = [
        ("pointwise_mlp", PointwiseMLP(in_vars=7, num_depths=num_depths, hidden_dim=64), "Pointwise MLP"),
        ("simple_cnn", SimpleCNNBaseline(in_vars=7, num_depths=num_depths, hidden_dim=64), "Simple CNN Baseline")
    ]

    for model_key, model, display_name in trainable_baselines:
        print(f"\n" + "=" * 70)
        print(f"TRAINING {display_name.upper()} ({model_key}) — {epochs} EPOCHS")
        print("=" * 70)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ckpt_path = f"checkpoints/{model_key}_best.pt"
        latest_path = f"checkpoints/{model_key}_latest.pt"

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_loss = float("inf")
        history = []
        t_train_start = time.time()

        for epoch in range(1, epochs + 1):
            ep_t0 = time.time()
            model.train()
            train_loss = 0.0
            n_batches = 0

            for x_b, y_b, _ in train_loader:
                optimizer.zero_grad()
                pred = model(x_b, depth_indices)
                loss = loss_fn(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(1, n_batches)

            # Validation loss check
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
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}, ckpt_path)

            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss}, latest_path)

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "lr": round(current_lr, 6),
                "epoch_time_s": round(ep_time, 2),
                "is_best": is_best
            })

            print(f"  Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {'*** BEST' if is_best else ''}")

        total_train_time = time.time() - t_train_start
        print(f"\nTraining completed in {total_train_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

        # Load best checkpoint for full out-of-sample evaluation
        best_ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {best_ckpt['epoch']} for final evaluation.")

        # Full Validation Set Evaluation
        val_eval = evaluate_model_full(model, val_ds, depth_values, batch_size=batch_size)
        print(f"\n  [August 2020 Validation Set]")
        print(f"    RMSE: {val_eval['overall']['rmse']:.4f}°C | MAE: {val_eval['overall']['mae']:.4f}°C | Bias: {val_eval['overall']['bias']:+.4f}°C | Corr: {val_eval['overall']['corr']:.4f}")

        # Full Test Set Evaluation (September 2020)
        test_eval = evaluate_model_full(model, test_ds, depth_values, batch_size=batch_size)
        print(f"\n  [September 2020 Test Set]")
        print(f"    RMSE: {test_eval['overall']['rmse']:.4f}°C | MAE: {test_eval['overall']['mae']:.4f}°C | Bias: {test_eval['overall']['bias']:+.4f}°C | Corr: {test_eval['overall']['corr']:.4f}")

        results["models"][model_key] = {
            "display_name": display_name,
            "parameters": param_count,
            "training_duration_s": round(total_train_time, 2),
            "best_epoch": best_ckpt["epoch"],
            "best_val_loss": round(best_val_loss, 4),
            "checkpoint": ckpt_path,
            "val_metrics": val_eval,
            "test_metrics": test_eval,
            "history": history
        }

    # Save JSON results
    out_json = "reports/results/baseline_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline results to {out_json}")

    # Generate Markdown Report
    report_md = "docs/BASELINE_RESULTS.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# SIH26066 — Phase 4 Stage 5 Baseline Models Evaluation Report\n\n")
        f.write(f"**Generated**: {results['timestamp']}  \n")
        f.write(f"**Training Period**: {results['train_period']['start']} to {results['train_period']['end']} ({results['train_period']['num_days']} days)  \n")
        f.write(f"**Validation Period**: {results['val_period']['start']} to {results['val_period']['end']} ({results['val_period']['num_days']} days)  \n")
        f.write(f"**Test Period (Held-Out)**: {results['test_period']['start']} to {results['test_period']['end']} ({results['test_period']['num_days']} days)  \n")
        f.write(f"**Scaler Used**: `{results['scaler_used']}` (strictly fitted on Jan–Jul 2020 training data)\n\n")

        f.write("## 1. Overall Performance Comparison (Test Set: September 2020)\n\n")
        f.write("| Model | Parameters | Test RMSE (°C) | Test MAE (°C) | Test Bias (°C) | Pearson r |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for k, m in results["models"].items():
            tm = m["test_metrics"]["overall"]
            f.write(f"| **{m['display_name']}** | {m['parameters']:,} | **{tm['rmse']:.4f}** | {tm['mae']:.4f} | {tm['bias']:+.4f} | {tm['corr']:.4f} |\n")
        f.write("\n")

        f.write("## 2. Depth-Stratified Performance on September 2020 Test Set\n\n")
        f.write("| Depth | Climatology RMSE | PointwiseMLP RMSE | SimpleCNN RMSE | SimpleCNN MAE | SimpleCNN Corr |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for d in depth_values:
            d_key = f"{int(d)}m"
            c_rmse = results["models"]["climatology"]["test_metrics"]["depth_wise"][d_key]["rmse"]
            mlp_rmse = results["models"]["pointwise_mlp"]["test_metrics"]["depth_wise"][d_key]["rmse"]
            cnn_rmse = results["models"]["simple_cnn"]["test_metrics"]["depth_wise"][d_key]["rmse"]
            cnn_mae = results["models"]["simple_cnn"]["test_metrics"]["depth_wise"][d_key]["mae"]
            cnn_corr = results["models"]["simple_cnn"]["test_metrics"]["depth_wise"][d_key]["corr"]
            f.write(f"| **{int(d)} m** | {c_rmse:.4f}°C | {mlp_rmse:.4f}°C | **{cnn_rmse:.4f}°C** | {cnn_mae:.4f}°C | {cnn_corr:.4f} |\n")
        f.write("\n")

        f.write("## 3. Training Dynamics Summary\n\n")
        for k in ["pointwise_mlp", "simple_cnn"]:
            m = results["models"][k]
            f.write(f"### {m['display_name']}\n")
            f.write(f"- Total Training Time: {m['training_duration_s']:.1f}s ({epochs} epochs)\n")
            f.write(f"- Best Epoch: {m['best_epoch']} (Val Loss: {m['best_val_loss']:.4f})\n")
            f.write(f"- Best Checkpoint: `{m['checkpoint']}`\n\n")

    print(f"Generated human-readable report at {report_md}")


if __name__ == "__main__":
    main()
