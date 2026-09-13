"""
SIH26066 — Stage 1 Training Readiness Audit
Comprehensive audit of Jan-Sep 2020 dataset, data quality, chunk loading,
models (forward/backward/optimizer/checkpoint), and hardware.
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
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath("."))
from src.data.catalog import TARGET_GRID, TARGET_DEPTHS, SURFACE_CHANNELS
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.models.oceanembed import OceanEmbedNet
from src.training.loss import MaskedMSELoss


def audit_hardware():
    print("\n--- 1. HARDWARE AUDIT ---")
    cuda_avail = torch.cuda.is_available()
    hw_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "device": "cuda" if cuda_avail else "cpu",
        "cpu_count_logical": os.cpu_count(),
    }
    
    vm = psutil.virtual_memory()
    hw_info["ram_total_gb"] = round(vm.total / (1024**3), 2)
    hw_info["ram_available_gb"] = round(vm.available / (1024**3), 2)
    hw_info["ram_used_pct"] = vm.percent

    if cuda_avail:
        hw_info["gpu_name"] = torch.cuda.get_device_name(0)
        hw_info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"  CUDA: Available | Device: {hw_info['gpu_name']} | VRAM: {hw_info['gpu_vram_gb']} GB")
    else:
        print(f"  CUDA: NOT Available in PyTorch | Falling back to CPU")
    print(f"  RAM: {hw_info['ram_available_gb']} GB free / {hw_info['ram_total_gb']} GB total ({hw_info['ram_used_pct']}% used)")
    print(f"  CPUs: {hw_info['cpu_count_logical']} logical cores")
    
    return hw_info


def audit_dataset_and_quality(processed_dir="data/processed"):
    print("\n--- 2. DATASET & QUALITY AUDIT (Jan-Sep 2020) ---")
    chunk_files = [f"chunk_2020_{m:02d}.pt" for m in range(1, 10)]
    
    audit_res = {
        "chunks": {},
        "dates": [],
        "expected_dates": 274,
        "actual_dates": 0,
        "date_continuity": False,
        "shapes_valid": True,
        "channel_stats": {},
        "target_stats": {},
        "land_ocean_mask_valid": True,
        "anomalies_detected": []
    }

    # Expected date range: 2020-01-01 to 2020-09-30 (leap year: Feb has 29 days)
    # Jan: 31, Feb: 29, Mar: 31, Apr: 30, May: 31, Jun: 30, Jul: 31, Aug: 31, Sep: 30 = 274 days
    start_d = datetime(2020, 1, 1)
    expected_date_list = [(start_d + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(274)]

    all_dates = []
    
    # Running accumulators for physical variables across valid ocean pixels
    num_channels = 7
    channel_names = [c[0] for c in SURFACE_CHANNELS]
    counts = np.zeros(num_channels, dtype=np.int64)
    total_pixels = np.zeros(num_channels, dtype=np.int64)
    min_vals = np.full(num_channels, np.inf)
    max_vals = np.full(num_channels, -np.inf)
    sum_vals = np.zeros(num_channels, dtype=np.float64)
    sum_sq_vals = np.zeros(num_channels, dtype=np.float64)

    # Target stats across 15 depths
    num_depths = len(TARGET_DEPTHS.depths)
    t_counts = np.zeros(num_depths, dtype=np.int64)
    t_total_pixels = np.zeros(num_depths, dtype=np.int64)
    t_min = np.full(num_depths, np.inf)
    t_max = np.full(num_depths, -np.inf)
    t_sum = np.zeros(num_depths, dtype=np.float64)
    t_sum_sq = np.zeros(num_depths, dtype=np.float64)

    first_ocean_mask = None

    for cf in chunk_files:
        cpath = os.path.join(processed_dir, cf)
        if not os.path.exists(cpath):
            audit_res["anomalies_detected"].append(f"Missing chunk: {cf}")
            print(f"  ERROR: {cf} does not exist!")
            continue

        data = torch.load(cpath, weights_only=False)
        X = data["X"]  # [N, 14, 101, 241]
        Y = data["Y"]  # [N, 15, 101, 241]
        dates = data["dates"]
        N = X.shape[0]

        all_dates.extend(dates)

        # Check shapes
        if list(X.shape[1:]) != [14, 101, 241]:
            audit_res["shapes_valid"] = False
            audit_res["anomalies_detected"].append(f"{cf}: Unexpected X shape {list(X.shape)}")
        if list(Y.shape[1:]) != [15, 101, 241]:
            audit_res["shapes_valid"] = False
            audit_res["anomalies_detected"].append(f"{cf}: Unexpected Y shape {list(Y.shape)}")

        # Check ocean mask consistency
        # channel 7 is SST mask (1.0 = ocean, 0.0 = land/missing)
        sst_mask = X[:, 7].numpy()
        if not np.all((sst_mask == 0.0) | (sst_mask == 1.0)):
            audit_res["land_ocean_mask_valid"] = False
            audit_res["anomalies_detected"].append(f"{cf}: Non-binary mask values in SST mask")

        if first_ocean_mask is None:
            first_ocean_mask = sst_mask[0]
        else:
            # Check if land geometry is consistent
            diff = np.sum(first_ocean_mask != sst_mask[0])
            if diff > 100:
                audit_res["anomalies_detected"].append(f"{cf}: Ocean mask differs from Jan by {diff} pixels")

        # Accumulate stats for surface variables
        X_np = X.numpy()
        for c in range(num_channels):
            var_data = X_np[:, c]
            mask_data = X_np[:, c + num_channels]
            total_pixels[c] += var_data.size
            valid = (mask_data == 1.0) & (~np.isnan(var_data)) & (~np.isinf(var_data))
            pix = var_data[valid].astype(np.float64)
            n_pix = len(pix)
            counts[c] += n_pix
            if n_pix > 0:
                min_vals[c] = min(min_vals[c], float(np.min(pix)))
                max_vals[c] = max(max_vals[c], float(np.max(pix)))
                sum_vals[c] += np.sum(pix)
                sum_sq_vals[c] += np.sum(pix**2)

        # Accumulate stats for targets Y across depths
        Y_np = Y.numpy()
        for d in range(num_depths):
            depth_data = Y_np[:, d]
            t_total_pixels[d] += depth_data.size
            valid_t = (~np.isnan(depth_data)) & (~np.isinf(depth_data)) & (depth_data > -5.0) & (depth_data < 45.0)
            t_pix = depth_data[valid_t].astype(np.float64)
            n_t = len(t_pix)
            t_counts[d] += n_t
            if n_t > 0:
                t_min[d] = min(t_min[d], float(np.min(t_pix)))
                t_max[d] = max(t_max[d], float(np.max(t_pix)))
                t_sum[d] += np.sum(t_pix)
                t_sum_sq[d] += np.sum(t_pix**2)

        audit_res["chunks"][cf] = {
            "days": N,
            "start": dates[0],
            "end": dates[-1],
            "X_shape": list(X.shape),
            "Y_shape": list(Y.shape),
            "size_mb": round(os.path.getsize(cpath) / (1024*1024), 2)
        }
        del data, X, Y, X_np, Y_np
        gc.collect()

    audit_res["actual_dates"] = len(all_dates)
    audit_res["dates"] = all_dates

    # Check date continuity
    missing_dates = set(expected_date_list) - set(all_dates)
    extra_dates = set(all_dates) - set(expected_date_list)
    duplicates = len(all_dates) - len(set(all_dates))
    is_sorted = (all_dates == sorted(all_dates))

    audit_res["date_continuity"] = (len(missing_dates) == 0 and duplicates == 0 and is_sorted)
    audit_res["missing_dates"] = sorted(list(missing_dates))
    audit_res["duplicate_dates_count"] = duplicates
    audit_res["chronologically_sorted"] = is_sorted

    print(f"  Total days found: {len(all_dates)} / expected: {audit_res['expected_dates']}")
    print(f"  Date range: {all_dates[0]} to {all_dates[-1]}")
    print(f"  Chronologically sorted: {is_sorted} | Duplicates: {duplicates} | Missing: {len(missing_dates)}")

    # Surface channel summary
    print("\n  Surface Input Variables (0.25° NIO Grid: 101x241):")
    print(f"  {'Channel':<10} | {'Name':<22} | {'Valid Frac':<10} | {'Min':<8} | {'Max':<8} | {'Mean':<8} | {'Std':<8} | {'Status'}")
    print("  " + "-" * 95)
    for c in range(num_channels):
        val_frac = float(counts[c] / max(1, total_pixels[c]))
        mean_v = float(sum_vals[c] / max(1, counts[c]))
        var_v = float((sum_sq_vals[c] / max(1, counts[c])) - (mean_v**2))
        std_v = float(np.sqrt(max(0.0, var_v)))
        
        status = "OK"
        if val_frac < 0.1:
            status = "CRITICAL_LOW_COVERAGE"
            audit_res["anomalies_detected"].append(f"Channel {c} ({channel_names[c]}): valid fraction < 0.1")
        elif std_v < 1e-4:
            status = "SUSPICIOUS_CONSTANT"
            audit_res["anomalies_detected"].append(f"Channel {c} ({channel_names[c]}): near-constant std={std_v}")
            
        audit_res["channel_stats"][channel_names[c]] = {
            "channel_idx": c,
            "valid_fraction": round(val_frac, 4),
            "min": round(float(min_vals[c]), 4),
            "max": round(float(max_vals[c]), 4),
            "mean": round(mean_v, 4),
            "std": round(std_v, 4),
            "status": status
        }
        print(f"  Ch {c:<7} | {channel_names[c]:<22} | {val_frac*100:<9.2f}% | {min_vals[c]:<8.2f} | {max_vals[c]:<8.2f} | {mean_v:<8.2f} | {std_v:<8.2f} | {status}")

    # Target depths summary
    print("\n  Target Depths (GLORYS Reference, 15 Depths):")
    print(f"  {'Depth':<8} | {'Valid Frac':<10} | {'Min (°C)':<10} | {'Max (°C)':<10} | {'Mean (°C)':<10} | {'Std (°C)':<10}")
    print("  " + "-" * 75)
    for d, depth_m in enumerate(TARGET_DEPTHS.depths):
        val_frac = float(t_counts[d] / max(1, t_total_pixels[d]))
        mean_v = float(t_sum[d] / max(1, t_counts[d]))
        var_v = float((t_sum_sq[d] / max(1, t_counts[d])) - (mean_v**2))
        std_v = float(np.sqrt(max(0.0, var_v)))

        audit_res["target_stats"][f"{int(depth_m)}m"] = {
            "depth_m": float(depth_m),
            "valid_fraction": round(val_frac, 4),
            "min": round(float(t_min[d]), 4),
            "max": round(float(t_max[d]), 4),
            "mean": round(mean_v, 4),
            "std": round(std_v, 4)
        }
        print(f"  {int(depth_m):<4} m   | {val_frac*100:<9.2f}% | {t_min[d]:<10.2f} | {t_max[d]:<10.2f} | {mean_v:<10.2f} | {std_v:<10.2f}")

    return audit_res


def audit_dataloader_and_streaming(processed_dir="data/processed"):
    print("\n--- 3. DATA LOADING & STREAMING AUDIT ---")
    chunk_files = [f"chunk_2020_{m:02d}.pt" for m in range(1, 4)]  # Test first 3 chunks
    
    class LazyDataset(Dataset):
        def __init__(self, pdir, cfiles):
            self.pdir = pdir
            self.index = []
            for cf in cfiles:
                cp = os.path.join(pdir, cf)
                d = torch.load(cp, weights_only=False)
                for i in range(d["X"].shape[0]):
                    self.index.append((cf, i, d["dates"][i]))
                del d
            self.cached_file = None
            self.cached_data = None

        def __len__(self):
            return len(self.index)

        def __getitem__(self, idx):
            cf, offset, dt = self.index[idx]
            if self.cached_file != cf:
                self.cached_file = cf
                self.cached_data = torch.load(os.path.join(self.pdir, cf), weights_only=False)
            return self.cached_data["X"][offset], self.cached_data["Y"][offset], dt

    ds = LazyDataset(processed_dir, chunk_files)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    t0 = time.time()
    batch_count = 0
    shapes_ok = True
    for x_b, y_b, dts in loader:
        batch_count += 1
        if x_b.shape != torch.Size([4, 14, 101, 241]) or y_b.shape != torch.Size([4, 15, 101, 241]):
            shapes_ok = False
        if batch_count >= 5:
            break
            
    elapsed = time.time() - t0
    print(f"  Loaded 5 batches of size 4 in {elapsed:.2f}s ({elapsed/5:.4f}s/batch)")
    print(f"  Batch X shape: {x_b.shape} | Batch Y shape: {y_b.shape} | Shapes valid: {shapes_ok}")
    return {"status": "PASSED" if shapes_ok else "FAILED", "sec_per_batch": round(elapsed/5, 4)}


def audit_models_forward_backward_optimizer():
    print("\n--- 4. MODEL FORWARD/BACKWARD/CHECKPOINT AUDIT ---")
    B, C_in, num_depths, H, W = 2, 14, 15, 101, 241
    depth_indices = torch.arange(num_depths)
    x = torch.randn(B, C_in, H, W)
    y = torch.randn(B, num_depths, H, W)
    loss_fn = MaskedMSELoss()

    models = [
        ("PointwiseMLP", PointwiseMLP(in_vars=7, num_depths=num_depths, hidden_dim=64)),
        ("SimpleCNNBaseline", SimpleCNNBaseline(in_vars=7, num_depths=num_depths, hidden_dim=64)),
        ("OceanEmbedNet", OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128))
    ]

    os.makedirs("checkpoints/audit_test", exist_ok=True)
    results = {}

    for name, model in models:
        print(f"\n  Auditing {name}:")
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Parameters: {param_count:,}")

        # Forward pass
        t0 = time.time()
        pred = model(x, depth_indices)
        fwd_time = time.time() - t0
        assert pred.shape == (B, num_depths, H, W), f"Expected shape {(B, num_depths, H, W)}, got {pred.shape}"
        print(f"    Forward pass: SUCCESS ({fwd_time:.3f}s, output shape {list(pred.shape)})")

        # Loss calculation
        loss = loss_fn(pred, y)
        assert torch.isfinite(loss).item(), "Loss is NaN or Inf!"
        print(f"    Loss calculation: SUCCESS (Loss = {loss.item():.4f})")

        # Backward pass
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        t0 = time.time()
        loss.backward()
        bwd_time = time.time() - t0
        
        # Check gradients are finite
        grads_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        assert grads_finite, "NaN or Inf detected in gradients!"
        print(f"    Backward pass: SUCCESS ({bwd_time:.3f}s, all gradients finite)")

        # Optimizer step
        optimizer.step()
        print("    Optimizer step: SUCCESS")

        # Checkpoint saving & loading
        ckpt_file = f"checkpoints/audit_test/{name}_test.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_file)
        assert os.path.exists(ckpt_file), "Checkpoint file was not created!"
        ckpt_size_kb = os.path.getsize(ckpt_file) / 1024
        
        # Reload
        loaded = torch.load(ckpt_file, weights_only=True)
        model.load_state_dict(loaded["model_state_dict"])
        print(f"    Checkpoint Save/Reload: SUCCESS ({ckpt_size_kb:.1f} KB)")

        results[name] = {
            "parameters": param_count,
            "forward_time_s": round(fwd_time, 4),
            "backward_time_s": round(bwd_time, 4),
            "checkpoint_kb": round(ckpt_size_kb, 1),
            "status": "PASSED"
        }

    # Clean up test checkpoints
    for name, _ in models:
        cf = f"checkpoints/audit_test/{name}_test.pt"
        if os.path.exists(cf):
            os.remove(cf)
    if os.path.exists("checkpoints/audit_test"):
        try:
            os.rmdir("checkpoints/audit_test")
        except Exception:
            pass

    return results


def main():
    print("=" * 80)
    print("SIH26066 — PHASE 4 STAGE 1: FINAL TRAINING-READINESS AUDIT")
    print("=" * 80)

    hw = audit_hardware()
    data_audit = audit_dataset_and_quality()
    streaming_audit = audit_dataloader_and_streaming()
    model_audit = audit_models_forward_backward_optimizer()

    # Determine readiness
    is_ready = (
        data_audit["actual_dates"] == 274 and
        data_audit["date_continuity"] and
        data_audit["shapes_valid"] and
        len(data_audit["anomalies_detected"]) == 0 and
        streaming_audit["status"] == "PASSED" and
        all(m["status"] == "PASSED" for m in model_audit.values())
    )

    full_report = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "STAGE 1: FINAL TRAINING-READINESS AUDIT",
        "training_readiness_status": "PASSED" if is_ready else "BLOCKED",
        "hardware": hw,
        "dataset_audit": {
            "expected_days": data_audit["expected_dates"],
            "actual_days": data_audit["actual_dates"],
            "date_start": data_audit["dates"][0],
            "date_end": data_audit["dates"][-1],
            "date_continuity": data_audit["date_continuity"],
            "missing_dates": data_audit["missing_dates"],
            "duplicate_dates_count": data_audit["duplicate_dates_count"],
            "shapes_valid": data_audit["shapes_valid"],
            "land_ocean_mask_valid": data_audit["land_ocean_mask_valid"],
            "anomalies": data_audit["anomalies_detected"],
            "channel_stats": data_audit["channel_stats"],
            "target_stats": data_audit["target_stats"]
        },
        "streaming_audit": streaming_audit,
        "model_audit": model_audit
    }

    os.makedirs("reports/real", exist_ok=True)
    report_file = "reports/real/stage1_readiness_audit.json"
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nAudit complete! Saved full audit report to: {report_file}")
    print(f"OVERALL TRAINING-READINESS STATUS: {'PASSED (READY TO TRAIN)' if is_ready else 'BLOCKED'}")


if __name__ == "__main__":
    main()
