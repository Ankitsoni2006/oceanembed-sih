"""
SIH26066 — Phase 4 Stage 11: Inference Efficiency & Computational Resource Benchmark
Measures actual per-day inference latency, throughput, peak memory, parameter count,
and checkpoint footprint across PointwiseMLP, SimpleCNNBaseline, and OceanEmbedNet.
Only reports empirically measured values.
"""

import os
import sys
import gc
import json
import time
import psutil
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.models.oceanembed import OceanEmbedNet
from src.data.catalog import TARGET_DEPTHS, TARGET_GRID
from src.preprocessing.normalization import OceanStandardScaler


def benchmark_model_inference():
    print("=" * 80)
    print("SIH26066 — PHASE 4 STAGE 11: INFERENCE EFFICIENCY BENCHMARK")
    print("=" * 80)

    depth_values = list(TARGET_DEPTHS.depths)
    num_depths = len(depth_values)
    depth_indices = torch.arange(num_depths)

    scaler_path = "configs/scaler_params_experiment_2020.json"
    scaler = OceanStandardScaler.load(scaler_path)

    # 1 daily NIO grid input: [1, 14, 101, 241]
    test_chunk = torch.load("data/processed/chunk_2020_09.pt", weights_only=False)
    raw_daily_grid = test_chunk["X"][0:1]  # Day 1 of September 2020
    x_input = scaler.transform(raw_daily_grid)

    models_config = [
        ("pointwise_mlp", "checkpoints/pointwise_mlp_best.pt", PointwiseMLP(in_vars=7, num_depths=num_depths), "Pointwise MLP"),
        ("simple_cnn", "checkpoints/simple_cnn_best.pt", SimpleCNNBaseline(in_vars=7, num_depths=num_depths), "Simple CNN Baseline"),
        ("oceanembed", "checkpoints/oceanembed_best.pt", OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128), "OceanEmbedNet (Multi-Scale U-Net)")
    ]

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_grid_dimensions": [1, 14, TARGET_GRID.shape[0], TARGET_GRID.shape[1]],
        "target_depths_count": num_depths,
        "models": {}
    }

    print(f"{'Model Architecture':<35} | {'Params':<10} | {'Latency (ms)':<16} | {'Throughput':<16} | {'Checkpoint (MB)':<16}")
    print("-" * 105)

    for model_key, ckpt_path, model, display_name in models_config:
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for {model_key} at {ckpt_path}. Skipping...")
            continue

        ckpt_data = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        ckpt_bytes = os.path.getsize(ckpt_path)
        ckpt_mb = ckpt_bytes / (1024 * 1024)

        # Warm-up (3 runs)
        with torch.no_grad():
            for _ in range(3):
                _ = model(x_input, depth_indices)

        # Timed benchmark: 20 repetitions
        latencies_ms = []
        process = psutil.Process()
        ram_before_mb = process.memory_info().rss / (1024 * 1024)

        with torch.no_grad():
            for _ in range(20):
                t0 = time.perf_counter()
                out = model(x_input, depth_indices)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        ram_after_mb = process.memory_info().rss / (1024 * 1024)
        mean_lat_ms = float(np.mean(latencies_ms))
        std_lat_ms = float(np.std(latencies_ms))
        throughput_grids_per_sec = 1000.0 / mean_lat_ms

        print(f"{display_name:<35} | {param_count:<10,d} | {mean_lat_ms:>7.2f} ± {std_lat_ms:<6.2f} | {throughput_grids_per_sec:>6.2f} grids/s | {ckpt_mb:>7.2f} MB")

        results["models"][model_key] = {
            "display_name": display_name,
            "parameter_count": param_count,
            "checkpoint_size_bytes": ckpt_bytes,
            "checkpoint_size_mb": round(ckpt_mb, 2),
            "inference_latency_ms": {
                "mean": round(mean_lat_ms, 2),
                "std": round(std_lat_ms, 2),
                "min": round(float(np.min(latencies_ms)), 2),
                "max": round(float(np.max(latencies_ms)), 2)
            },
            "throughput_grids_per_sec": round(throughput_grids_per_sec, 2),
            "ram_memory_rss_mb": round(ram_after_mb, 1)
        }

    out_json = "reports/results/inference_efficiency_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved inference efficiency benchmark to: {out_json}")


if __name__ == "__main__":
    benchmark_model_inference()
