# SIH26066 — Phase 4 Stage 5 Baseline Models Evaluation Report

**Generated**: 2026-09-13T06:31:40Z  
**Training Period**: 2020-01-01 to 2020-07-31 (213 days)  
**Validation Period**: 2020-08-01 to 2020-08-31 (31 days)  
**Test Period (Held-Out)**: 2020-09-01 to 2020-09-30 (30 days)  
**Scaler Used**: `configs/scaler_params_experiment_2020.json` (strictly fitted on Jan–Jul 2020 training data)

## 1. Overall Performance Comparison (Test Set: September 2020)

| Model | Parameters | Test RMSE (°C) | Test MAE (°C) | Test Bias (°C) | Pearson r |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Climatology Profile** | 0 | **2.9976** | 2.4705 | -2.1353 | 0.9663 |
| **Pointwise MLP** | 6,095 | **1.1177** | 0.7569 | +0.0547 | 0.9893 |
| **Simple CNN Baseline** | 46,031 | **1.0418** | 0.7100 | -0.0052 | 0.9907 |

## 2. Depth-Stratified Performance on September 2020 Test Set

| Depth | Climatology RMSE | PointwiseMLP RMSE | SimpleCNN RMSE | SimpleCNN MAE | SimpleCNN Corr |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0 m** | 1.6988°C | 0.5800°C | **0.5034°C** | 0.3579°C | 0.9467 |
| **5 m** | 1.7085°C | 0.6263°C | **0.5380°C** | 0.3776°C | 0.9390 |
| **10 m** | 1.7406°C | 0.6613°C | **0.5897°C** | 0.3963°C | 0.9258 |
| **20 m** | 1.8716°C | 0.9074°C | **0.8719°C** | 0.5394°C | 0.8440 |
| **30 m** | 2.1576°C | 1.1185°C | **1.0148°C** | 0.6546°C | 0.8082 |
| **50 m** | 3.0670°C | 1.4708°C | **1.2576°C** | 0.9252°C | 0.8333 |
| **75 m** | 3.7585°C | 1.6939°C | **1.5454°C** | 1.1967°C | 0.7956 |
| **100 m** | 4.1258°C | 1.6020°C | **1.5529°C** | 1.2033°C | 0.7769 |
| **125 m** | 4.2860°C | 1.4579°C | **1.4665°C** | 1.1321°C | 0.7915 |
| **150 m** | 4.2128°C | 1.4221°C | **1.3897°C** | 1.0721°C | 0.8056 |
| **200 m** | 3.3741°C | 1.2936°C | **1.2454°C** | 0.9617°C | 0.7757 |
| **300 m** | 2.9251°C | 1.0199°C | **0.8979°C** | 0.6631°C | 0.7963 |
| **500 m** | 3.0636°C | 0.7162°C | **0.6317°C** | 0.4626°C | 0.8502 |
| **700 m** | 3.0362°C | 0.6902°C | **0.6200°C** | 0.4466°C | 0.8343 |
| **1000 m** | 2.7339°C | 0.6389°C | **0.5946°C** | 0.4303°C | 0.7968 |

## 3. Training Dynamics Summary

### Pointwise MLP
- Total Training Time: 77.0s (15 epochs)
- Best Epoch: 13 (Val Loss: 1.1782)
- Best Checkpoint: `checkpoints/pointwise_mlp_best.pt`

### Simple CNN Baseline
- Total Training Time: 152.9s (15 epochs)
- Best Epoch: 10 (Val Loss: 0.9526)
- Best Checkpoint: `checkpoints/simple_cnn_best.pt`

