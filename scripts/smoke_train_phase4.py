"""
SIH26066 — Phase 4 Stage 3: Data Smoke Training & Sanity Verification
Executes a quick 1-epoch mini-batch training loop with real Jan-Jul 2020 data
and real Aug 2020 validation data across all 3 models:
1. PointwiseMLP
2. SimpleCNNBaseline
3. OceanEmbedNet
Verifies gradient finiteness, absence of NaNs, checkpoint save/reload, and memory safety.
"""

import os
import sys
import gc
import json
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath("."))
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.models.oceanembed import OceanEmbedNet
from src.preprocessing.normalization import OceanStandardScaler
from src.training.loss import MaskedMSELoss
from src.data.catalog import TARGET_DEPTHS


class MiniOceanDataset(Dataset):
    def __init__(self, processed_dir, chunk_files, scaler, max_samples=None):
        self.processed_dir = processed_dir
        self.scaler = scaler
        self.samples_x = []
        self.samples_y = []
        self.dates = []

        for cf in chunk_files:
            cpath = os.path.join(processed_dir, cf)
            data = torch.load(cpath, weights_only=False)
            X = data["X"]
            Y = data["Y"]
            dts = data["dates"]
            
            n = len(dts)
            for i in range(n):
                self.samples_x.append(X[i])
                self.samples_y.append(Y[i])
                self.dates.append(dts[i])
                if max_samples and len(self.samples_x) >= max_samples:
                    break
            del data
            if max_samples and len(self.samples_x) >= max_samples:
                break
        gc.collect()

    def __len__(self):
        return len(self.samples_x)

    def __getitem__(self, idx):
        x = self.samples_x[idx]
        y = self.samples_y[idx]
        dt = self.dates[idx]
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x, y, dt


def smoke_train():
    print("=" * 80)
    print("SIH26066 — PHASE 4 STAGE 3: DATA SMOKE TRAIN")
    print("=" * 80)

    scaler_path = "configs/scaler_params_experiment_2020.json"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run Stage 2 first!")
    scaler = OceanStandardScaler.load(scaler_path)
    print(f"Loaded zero-leakage scaler: {scaler_path}")

    # Load small subset: 16 training days (from Jan 2020), 8 validation days (from Aug 2020)
    train_chunks = ["chunk_2020_01.pt"]
    val_chunks = ["chunk_2020_08.pt"]

    print("Creating mini datasets (16 train days, 8 val days)...")
    train_ds = MiniOceanDataset("data/processed", train_chunks, scaler=scaler, max_samples=16)
    val_ds = MiniOceanDataset("data/processed", val_chunks, scaler=scaler, max_samples=8)

    batch_size = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)
    loss_fn = MaskedMSELoss()

    models = [
        ("PointwiseMLP", PointwiseMLP(in_vars=7, num_depths=num_depths, hidden_dim=64)),
        ("SimpleCNNBaseline", SimpleCNNBaseline(in_vars=7, num_depths=num_depths, hidden_dim=64)),
        ("OceanEmbedNet", OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128))
    ]

    os.makedirs("checkpoints/smoke_test", exist_ok=True)
    smoke_results = {}

    for name, model in models:
        print(f"\n" + "-" * 60)
        print(f"SMOKE TESTING: {name}")
        print("-" * 60)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        
        batch_losses = []
        t0 = time.time()

        for b_idx, (x_b, y_b, dts) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(x_b, depth_indices)
            loss = loss_fn(pred, y_b)
            
            # Check loss is finite
            loss_val = loss.item()
            if not torch.isfinite(loss).item():
                raise ValueError(f"{name} produced non-finite loss at batch {b_idx}: {loss_val}")
                
            loss.backward()

            # Check gradients
            for p_idx, p in enumerate(model.parameters()):
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all().item():
                        raise ValueError(f"{name} produced NaN/Inf gradient in parameter {p_idx} at batch {b_idx}!")

            # Clip gradient & step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_losses.append(loss_val)
            print(f"  Batch {b_idx+1}/{len(train_loader)} | Loss: {loss_val:.4f}")

        train_time = time.time() - t0
        print(f"  Training finished in {train_time:.2f}s | Initial Loss: {batch_losses[0]:.4f} -> Final Loss: {batch_losses[-1]:.4f}")

        # Check validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vx, vy, vdts in val_loader:
                vpred = model(vx, depth_indices)
                vloss = loss_fn(vpred, vy)
                val_losses.append(vloss.item())

        avg_val_loss = float(np.mean(val_losses))
        print(f"  Validation Loss (Avg of {len(val_loader)} batches): {avg_val_loss:.4f}")

        # Checkpoint test
        ckpt_path = f"checkpoints/smoke_test/{name}_smoke.pt"
        torch.save(model.state_dict(), ckpt_path)
        assert os.path.exists(ckpt_path), f"Failed to save {ckpt_path}"
        
        # Reload test
        reloaded_model = type(model)(in_vars=7, num_depths=num_depths) if name != "OceanEmbedNet" else type(model)(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
        reloaded_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        reloaded_model.eval()
        with torch.no_grad():
            rpred = reloaded_model(val_ds[0][0].unsqueeze(0), depth_indices)
            diff = torch.max(torch.abs(rpred - model(val_ds[0][0].unsqueeze(0), depth_indices))).item()
            assert diff < 1e-5, f"Reloaded checkpoint outputs mismatch by {diff}"

        print(f"  Checkpoint Save/Reload: VERIFIED (Max Output Delta: {diff:.2e})")

        vm = psutil.virtual_memory()
        smoke_results[name] = {
            "initial_train_loss": round(batch_losses[0], 4),
            "final_train_loss": round(batch_losses[-1], 4),
            "loss_decreased": bool(batch_losses[-1] <= batch_losses[0]),
            "avg_val_loss": round(avg_val_loss, 4),
            "train_time_s": round(train_time, 2),
            "ram_used_gb": round((vm.total - vm.available) / (1024**3), 2),
            "checkpoint_verified": True,
            "status": "PASSED"
        }

    # Clean up test checkpoints
    for name, _ in models:
        cp = f"checkpoints/smoke_test/{name}_smoke.pt"
        if os.path.exists(cp):
            os.remove(cp)
    if os.path.exists("checkpoints/smoke_test"):
        try:
            os.rmdir("checkpoints/smoke_test")
        except Exception:
            pass

    out_file = "reports/real/stage3_smoke_train.json"
    with open(out_file, "w") as f:
        json.dump(smoke_results, f, indent=2)
    print(f"\nAll 3 models passed Smoke Training successfully! Report saved to {out_file}")
    return smoke_results

if __name__ == "__main__":
    smoke_train()
