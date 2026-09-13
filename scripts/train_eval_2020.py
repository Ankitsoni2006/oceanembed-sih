"""
SIH26066 — Full Year 2020 Model Training, Evaluation, and ARGO In-Situ Benchmarking
Trains Pointwise MLP, Simple CNN, and OceanEmbedNet on the 2020 training partition
(Jan 1 - Sep 30, 2020, 274 days), evaluates out-of-sample on Validation (Oct 1 - Nov 30, 61 days)
and Test (Dec 1 - Dec 31, 31 days), and validates against unassimilated ARGO floats.
"""

import os
import sys
import gc
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath("."))

from src.models.oceanembed import OceanEmbedNet
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.preprocessing.normalization import OceanStandardScaler
from src.training.loss import MaskedMSELoss
from src.data.catalog import TARGET_DEPTHS
from src.evaluation.metrics import calculate_metrics, evaluate_depth_wise as depth_stratified_metrics
from src.validation.argo_eval import ArgoValidator


class MultiChunkOceanDataset(Dataset):
    """
    Lazy dataset that loads monthly chunk files as needed without holding
    all chunks simultaneously in RAM. Uses an LRU cache of at most max_cached_chunks (~250MB)
    to eliminate disk thrashing during shuffled epoch iteration.
    """
    def __init__(self, processed_dir: str, chunk_files: List[str], scaler: Optional[OceanStandardScaler] = None, max_cached_chunks: int = 10):
        self.processed_dir = processed_dir
        self.chunk_files = chunk_files
        self.scaler = scaler
        self.max_cached_chunks = max_cached_chunks
        
        self.index = []
        self._cache = {}  # cf -> (X, Y)

        for cf in chunk_files:
            cpath = os.path.join(processed_dir, cf)
            if not os.path.exists(cpath):
                raise FileNotFoundError(f"Chunk file not found: {cpath}")
            data = torch.load(cpath, weights_only=False)
            num_days = data["X"].shape[0]
            dates = data["dates"]
            for offset in range(num_days):
                self.index.append((cf, offset, dates[offset]))
            del data
            gc.collect()

    def __len__(self):
        return len(self.index)

    def _get_chunk(self, cf: str):
        if cf in self._cache:
            return self._cache[cf]
        if len(self._cache) >= self.max_cached_chunks:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            gc.collect()
        cpath = os.path.join(self.processed_dir, cf)
        data = torch.load(cpath, weights_only=False)
        self._cache[cf] = (data["X"], data["Y"])
        return self._cache[cf]

    def __getitem__(self, idx: int):
        cf, offset, dt_str = self.index[idx]
        x_chunk, y_chunk = self._get_chunk(cf)
        
        x = x_chunk[offset] # [14, 101, 241]
        y = y_chunk[offset] # [15, 101, 241]
        
        if self.scaler is not None:
            x = self.scaler.transform(x)

        return x, y, dt_str


def compute_scaler_from_chunks(processed_dir: str, train_chunks: List[str], output_scaler_path: str) -> OceanStandardScaler:
    """Computes standard scaler parameters strictly over training chunks using running sums (RAM safe)."""
    print(f"\nComputing Zero-Leakage OceanStandardScaler across {len(train_chunks)} training chunks...")
    t0 = time.time()
    num_channels = 7
    total_counts = np.zeros(num_channels, dtype=np.int64)
    sum_vals = np.zeros(num_channels, dtype=np.float64)
    sum_sq_vals = np.zeros(num_channels, dtype=np.float64)

    for cf in train_chunks:
        cpath = os.path.join(processed_dir, cf)
        data = torch.load(cpath, weights_only=False)
        X_chunk = data["X"].numpy()
        for c in range(num_channels):
            var_data = X_chunk[:, c]
            mask_data = X_chunk[:, c + num_channels]
            valid = (mask_data == 1.0) & (~np.isnan(var_data)) & (~np.isinf(var_data))
            pixels = var_data[valid].astype(np.float64)
            total_counts[c] += len(pixels)
            sum_vals[c] += np.sum(pixels)
            sum_sq_vals[c] += np.sum(pixels ** 2)
        del data, X_chunk
        gc.collect()

    means = sum_vals / np.maximum(total_counts, 1)
    variances = (sum_sq_vals / np.maximum(total_counts, 1)) - (means ** 2)
    stds = np.sqrt(np.maximum(variances, 1e-8))

    scaler = OceanStandardScaler(num_physical_channels=num_channels)
    scaler.means = means.astype(np.float32)
    scaler.stds = stds.astype(np.float32)
    scaler.is_fitted = True
    scaler.save(output_scaler_path)

    print(f"Computed and saved scaler to {output_scaler_path} in {time.time()-t0:.2f}s:")
    for c in range(num_channels):
        print(f"  Channel {c}: Mean = {means[c]:.4f}, Std = {stds[c]:.4f} (N={total_counts[c]:,} valid pixels)")
    return scaler


def evaluate_model_on_dataset(model, dataset: MultiChunkOceanDataset, depth_values: List[float], batch_size: int = 4):
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

    # Overall metrics
    overall = calculate_metrics(cat_preds, cat_targets)
    # Depth-wise metrics
    depth_metrics = depth_stratified_metrics(cat_preds, cat_targets, depth_values)

    return {
        "overall": overall,
        "depth_wise": depth_metrics,
        "num_samples": len(all_dates),
        "date_start": all_dates[0],
        "date_end": all_dates[-1]
    }


def main():
    parser = argparse.ArgumentParser(description="SIH26066 — 2020 Model Training & Evaluation")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    processed_dir = "data/processed"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports/real", exist_ok=True)

    # 1. Define partitions
    train_chunks = [f"chunk_2020_{m:02d}.pt" for m in range(1, 10)]   # Jan - Sep (274 days)
    val_chunks   = [f"chunk_2020_{m:02d}.pt" for m in range(10, 12)]  # Oct - Nov (61 days)
    test_chunks  = [f"chunk_2020_12.pt"]                               # Dec (31 days)

    print("=" * 80)
    print("SIH26066 — FULL YEAR 2020 MULTI-MODEL BENCHMARK PIPELINE")
    print("=" * 80)
    print(f"Train Chunks ({len(train_chunks)}): {train_chunks}")
    print(f"Val Chunks   ({len(val_chunks)}):   {val_chunks}")
    print(f"Test Chunks  ({len(test_chunks)}):  {test_chunks}")

    # Verify all chunks exist
    all_chunks = train_chunks + val_chunks + test_chunks
    missing = [c for c in all_chunks if not os.path.exists(os.path.join(processed_dir, c))]
    if missing:
        print(f"\nERROR: Missing processed chunks: {missing}")
        print("Please run `python scripts/acquire_process_2020.py --months all` first.")
        sys.exit(1)

    # 2. Scaler
    scaler_path = "configs/scaler_params_2020.json"
    if not os.path.exists(scaler_path):
        scaler = compute_scaler_from_chunks(processed_dir, train_chunks, scaler_path)
    else:
        print(f"Loading existing scaler from {scaler_path}")
        scaler = OceanStandardScaler.load(scaler_path)

    # 3. Create Datasets
    print("\nInitializing datasets...")
    train_ds = MultiChunkOceanDataset(processed_dir, train_chunks, scaler=scaler)
    val_ds   = MultiChunkOceanDataset(processed_dir, val_chunks, scaler=scaler)
    test_ds  = MultiChunkOceanDataset(processed_dir, test_chunks, scaler=scaler)
    print(f"  Train samples: {len(train_ds):,} days ({train_ds.index[0][2]} to {train_ds.index[-1][2]})")
    print(f"  Val samples:   {len(val_ds):,} days ({val_ds.index[0][2]} to {val_ds.index[-1][2]})")
    print(f"  Test samples:  {len(test_ds):,} days ({test_ds.index[0][2]} to {test_ds.index[-1][2]})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)
    loss_fn = MaskedMSELoss()

    models_to_train = [
        ("mlp", PointwiseMLP(in_vars=7, num_depths=num_depths, hidden_dim=64), "Pointwise MLP"),
        ("simple_cnn", SimpleCNNBaseline(in_vars=7, num_depths=num_depths, hidden_dim=64), "Simple CNN Baseline"),
        ("oceanembed", OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128), "OceanEmbedNet")
    ]

    all_results = {
        "dataset_year": 2020,
        "train_period": {"start": train_ds.index[0][2], "end": train_ds.index[-1][2], "num_days": len(train_ds)},
        "val_period":   {"start": val_ds.index[0][2], "end": val_ds.index[-1][2], "num_days": len(val_ds)},
        "test_period":  {"start": test_ds.index[0][2], "end": test_ds.index[-1][2], "num_days": len(test_ds)},
        "scaler_used": scaler_path,
        "models": {}
    }

    for model_key, model, display_name in models_to_train:
        print(f"\n" + "=" * 70)
        print(f"TRAINING {display_name.upper()} ({model_key}) ON FULL YEAR 2020")
        print("=" * 70)

        ckpt_path = f"checkpoints/{model_key}_2020_best.pt"
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        best_val_loss = float("inf")
        history = []

        for epoch in range(1, args.epochs + 1):
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

            # Quick validation loss check
            model.eval()
            val_loss = 0.0
            v_batches = 0
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            with torch.no_grad():
                for x_v, y_v, _ in val_loader:
                    pred = model(x_v, depth_indices)
                    loss = loss_fn(pred, y_v)
                    val_loss += loss.item()
                    v_batches += 1
            val_loss /= max(1, v_batches)

            scheduler.step()
            ep_time = time.time() - ep_t0
            print(f"Epoch {epoch:02d}/{args.epochs:02d} [{ep_time:.1f}s] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_path)

        print(f"Best validation loss for {model_key}: {best_val_loss:.4f} (Saved to {ckpt_path})")

        # Load best checkpoint for rigorous evaluation
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        print(f"\n--- Evaluating {display_name} on Out-of-Sample Validation Set (Oct-Nov 2020) ---")
        val_metrics = evaluate_model_on_dataset(model, val_ds, depth_values, batch_size=args.batch_size)
        print(f"  Val RMSE: {val_metrics['overall']['rmse']:.4f}°C | MAE: {val_metrics['overall']['mae']:.4f}°C | Bias: {val_metrics['overall']['bias']:.4f}°C | Pearson r: {val_metrics['overall']['corr']:.4f}")

        print(f"\n--- Evaluating {display_name} on Out-of-Sample Test Set (Dec 2020) ---")
        test_metrics = evaluate_model_on_dataset(model, test_ds, depth_values, batch_size=args.batch_size)
        print(f"  Test RMSE: {test_metrics['overall']['rmse']:.4f}°C | MAE: {test_metrics['overall']['mae']:.4f}°C | Bias: {test_metrics['overall']['bias']:.4f}°C | Pearson r: {test_metrics['overall']['corr']:.4f}")

        # Independent ARGO Validation
        print(f"\n--- Evaluating {display_name} on Unassimilated ARGO Profiling Floats ---")
        argo_results = {}
        argo_candidates = [
            "data/raw/argo/argo_pilot_profiles.nc",
            "data/argo/20221101_prof.nc"
        ]
        argo_file = next((f for f in argo_candidates if os.path.exists(f)), None)
        if argo_file:
            try:
                validator = ArgoValidator(checkpoint_path=ckpt_path, model_type=model_key, scaler_path=scaler_path, argo_file=argo_file)
                # Use test sample
                raw_test = torch.load(os.path.join(processed_dir, test_chunks[0]), weights_only=False)
                argo_eval = validator.run_validation(raw_test["X"][:1])
                argo_results = argo_eval.get("metrics", {})
                print(f"  ARGO In-Situ RMSE: {argo_results.get('overall_rmse', float('nan')):.4f}°C (Profiles={argo_eval.get('num_profiles_matched')}, Points={argo_eval.get('total_colocated_points')})")
            except Exception as e:
                print(f"  ARGO validation error: {e}")

        all_results["models"][model_key] = {
            "name": display_name,
            "checkpoint": ckpt_path,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "argo_metrics": argo_results
        }

    # Save master benchmark JSON
    benchmark_json = "reports/real/2020_models_benchmark.json"
    with open(benchmark_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMaster 2020 benchmark JSON saved to: {benchmark_json}")


if __name__ == "__main__":
    main()
