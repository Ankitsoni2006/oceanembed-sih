"""
SIH26066 — Phase 4 Stage 4: Comprehensive Training Speed & Compute Benchmark
Measures batches/sec, sec/batch, RAM usage, and projects epoch times across 
PointwiseMLP, SimpleCNNBaseline, and OceanEmbedNet on the 213-day training partition.
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


class BenchmarkDataset(Dataset):
    def __init__(self, processed_dir, chunk_file, scaler, num_samples=20):
        self.scaler = scaler
        cpath = os.path.join(processed_dir, chunk_file)
        data = torch.load(cpath, weights_only=False)
        self.x = data["X"][:num_samples]
        self.y = data["Y"][:num_samples]
        self.dates = data["dates"][:num_samples]
        del data
        gc.collect()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x, y, self.dates[idx]


def benchmark_speed():
    print("=" * 80)
    print("SIH26066 — PHASE 4 STAGE 4: SPEED & RESOURCE BENCHMARK")
    print("=" * 80)

    scaler_path = "configs/scaler_params_experiment_2020.json"
    scaler = OceanStandardScaler.load(scaler_path)

    # 20 samples from Jan 2020
    ds = BenchmarkDataset("data/processed", "chunk_2020_01.pt", scaler=scaler, num_samples=20)

    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)
    loss_fn = MaskedMSELoss()

    total_train_samples = 213  # Jan 1 - Jul 31, 2020
    benchmark_results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_samples": total_train_samples,
        "models": {}
    }

    test_configs = [
        ("PointwiseMLP", PointwiseMLP(in_vars=7, num_depths=num_depths, hidden_dim=64), 4),
        ("SimpleCNNBaseline", SimpleCNNBaseline(in_vars=7, num_depths=num_depths, hidden_dim=64), 4),
        ("OceanEmbedNet", OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128), 2)
    ]

    for name, model, batch_size in test_configs:
        print(f"\nBenchmarking {name} (Batch Size = {batch_size})...")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        # Warm-up pass (1 batch)
        for x_w, y_w, _ in loader:
            optimizer.zero_grad()
            pred = model(x_w, depth_indices)
            loss = loss_fn(pred, y_w)
            loss.backward()
            optimizer.step()
            break

        # Timed benchmark pass (5 batches)
        num_bench_batches = 5
        timings = []
        process = psutil.Process()
        ram_before_mb = process.memory_info().rss / (1024 * 1024)

        b_count = 0
        t_bench_start = time.time()
        for x_b, y_b, _ in loader:
            t_b0 = time.time()
            optimizer.zero_grad()
            pred = model(x_b, depth_indices)
            loss = loss_fn(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            b_time = time.time() - t_b0
            timings.append(b_time)
            b_count += 1
            if b_count >= num_bench_batches:
                break

        total_elapsed = time.time() - t_bench_start
        ram_after_mb = process.memory_info().rss / (1024 * 1024)
        
        sec_per_batch = float(np.mean(timings))
        batches_per_sec = 1.0 / sec_per_batch
        samples_per_sec = batches_per_sec * batch_size
        
        batches_per_epoch = int(np.ceil(total_train_samples / batch_size))
        est_epoch_sec = batches_per_epoch * sec_per_batch
        est_epoch_min = est_epoch_sec / 60.0

        print(f"  Speed: {batches_per_sec:.2f} batches/s ({sec_per_batch:.3f} s/batch)")
        print(f"  Throughput: {samples_per_sec:.2f} samples/s")
        print(f"  Batches per Epoch (N={total_train_samples}): {batches_per_epoch}")
        print(f"  Estimated 1-Epoch Duration: {est_epoch_sec:.1f}s ({est_epoch_min:.2f} min)")
        print(f"  Process RAM: {ram_after_mb:.1f} MB (Delta: +{ram_after_mb - ram_before_mb:.1f} MB)")

        benchmark_results["models"][name] = {
            "batch_size": batch_size,
            "sec_per_batch": round(sec_per_batch, 4),
            "batches_per_sec": round(batches_per_sec, 2),
            "samples_per_sec": round(samples_per_sec, 2),
            "batches_per_epoch": batches_per_epoch,
            "estimated_epoch_sec": round(est_epoch_sec, 2),
            "estimated_epoch_min": round(est_epoch_min, 2),
            "ram_mb": round(ram_after_mb, 1)
        }

    # Propose training budget
    # PointwiseMLP: fast (~0.05s/batch -> ~2.7s/epoch) -> 15 epochs takes ~40 seconds
    # SimpleCNNBaseline: fast (~0.10s/batch -> ~5.4s/epoch) -> 15 epochs takes ~80 seconds
    # OceanEmbedNet: ~2.5s/batch (bs=2, 107 batches -> ~270s = 4.5 min/epoch)
    # 5 epochs of OceanEmbedNet takes ~22.5 minutes, 8 epochs takes ~36 minutes.
    # Let's see measured numbers first!

    os.makedirs("reports/real", exist_ok=True)
    out_json = "reports/real/stage4_speed_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved speed benchmark report to: {out_json}")


if __name__ == "__main__":
    benchmark_speed()
