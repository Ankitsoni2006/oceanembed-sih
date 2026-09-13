import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
from src.models.oceanembed import OceanEmbedNet
from src.training.loss import MaskedMSELoss

print("============================================================")
print("PHASE 13: EMPIRICAL COMPUTE BENCHMARK")
print("============================================================")

process = psutil.Process(os.getpid())
ram_start = process.memory_info().rss / (1024 * 1024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_start = torch.cuda.memory_allocated() / (1024 * 1024)
else:
    print("VRAM: N/A (CPU execution)")
    vram_start = 0.0

# Realistic batch size B=4
B = 4
H, W = 101, 241
in_channels = 14
out_channels = 15

X = torch.randn(B, in_channels, H, W, device=device)
Y = torch.randn(B, out_channels, H, W, device=device)

x_size_mb = (X.element_size() * X.nelement()) / (1024 * 1024)
y_size_mb = (Y.element_size() * Y.nelement()) / (1024 * 1024)

print(f"Batch size: {B}")
print(f"Input tensor shape: {tuple(X.shape)} -> Size: {x_size_mb:.2f} MB")
print(f"Output tensor shape: {tuple(Y.shape)} -> Size: {y_size_mb:.2f} MB")

# Instantiate model
model = OceanEmbedNet(in_vars=7, num_depths=15, base_features=32, embedding_dim=128).to(device)
criterion = MaskedMSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
depth_indices = torch.arange(15, device=device)

# Warmup pass
_ = model(X, depth_indices)

# Benchmark 5 iterations
bench_runs = 5
fwd_times = []
bwd_times = []
step_times = []

for _ in range(bench_runs):
    t0 = time.perf_counter()
    out = model(X, depth_indices)
    t1 = time.perf_counter()
    loss = criterion(out, Y)
    optimizer.zero_grad()
    loss.backward()
    t2 = time.perf_counter()
    optimizer.step()
    t3 = time.perf_counter()
    
    fwd_times.append((t1 - t0) * 1000)
    bwd_times.append((t2 - t1) * 1000)
    step_times.append((t3 - t0) * 1000)

ram_peak = process.memory_info().rss / (1024 * 1024)

mean_fwd = float(np.mean(fwd_times))
mean_bwd = float(np.mean(bwd_times))
mean_step = float(np.mean(step_times))

print(f"\n--- TIMING METRICS (Averaged over {bench_runs} iterations) ---")
print(f"Forward Pass Time:      {mean_fwd:7.2f} ms")
print(f"Backward Pass Time:     {mean_bwd:7.2f} ms")
print(f"Total Step Time:        {mean_step:7.2f} ms per batch (B={B})")
print(f"Throughput:             {B / (mean_step / 1000):7.2f} ocean-days / second")

print(f"\n--- MEMORY METRICS ---")
print(f"Process Base RAM:       {ram_start:7.2f} MB")
print(f"Process Peak RAM:       {ram_peak:7.2f} MB")
print(f"RAM Delta:              {ram_peak - ram_start:7.2f} MB")

# Projections based on measured benchmarks
days_per_year = 365
batches_per_year = days_per_year / B
sec_per_epoch_year = (batches_per_year * mean_step) / 1000
print(f"\n--- REAL TRAINING SCALE PROJECTIONS ---")
print(f"1 Year (365 days) 1 Epoch:     {sec_per_epoch_year:7.2f} s ({sec_per_epoch_year/60:4.2f} min)")
print(f"1 Year (365 days) 50 Epochs:   {sec_per_epoch_year*50/60:7.2f} min")
print(f"5 Years (1825 days) 50 Epochs: {sec_per_epoch_year*5*50/60:7.2f} min ({sec_per_epoch_year*5*50/3600:4.2f} hours)")
