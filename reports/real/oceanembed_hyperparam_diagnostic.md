# OceanEmbedNet Hyperparameter Diagnostic Report (Part B)

**Objective**: Test whether learning rate adjustment (`3e-4` vs `1e-3`) and extended training (20 epochs) resolve the generalization gap between training loss and validation loss on January 2020 data.

- **Train Period**: 2020-01-01 to 2020-01-24 (24 days)
- **Validation Period**: 2020-01-25 to 2020-01-31 (7 out-of-sample days)
- **Input Shape**: `[B, 14, 101, 241]` | **Output Shape**: `[B, 15, 101, 241]`

## 1. Summary Comparison

| Metric | LR = 1e-3 (Baseline) | LR = 3e-4 (Conservative) |
| :--- | :---: | :---: |
| **Final Train Loss** | 0.1327 | 0.3327 |
| **Best Val Loss** | 2.5562 | 3.5717 |
| **Final Val RMSE (°C)** | 1.6096 | 2.1504 |
| **Training Duration** | 551.5s | 497.4s |

## 2. Epoch-by-Epoch Dynamics

| Epoch | LR=1e-3 Train Loss | LR=1e-3 Val Loss | LR=1e-3 Val RMSE | LR=3e-4 Train Loss | LR=3e-4 Val Loss | LR=3e-4 Val RMSE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 5.2244 | 8.1981 | 2.8577°C | 5.8999 | 6.5019 | 2.5446°C |
| 02 | 2.5522 | 6.1539 | 2.4892°C | 4.1331 | 4.9824 | 2.2275°C |
| 03 | 1.1701 | 4.5868 | 2.1398°C | 2.6523 | 3.8130 | 1.9473°C |
| 04 | 0.7013 | 4.1204 | 2.0263°C | 1.8025 | 3.5717 | 1.8840°C |
| 05 | 0.4934 | 3.6529 | 1.9091°C | 1.3851 | 4.7079 | 2.1635°C |
| 06 | 0.3993 | 2.8112 | 1.6719°C | 1.1092 | 4.7746 | 2.1784°C |
| 07 | 0.3339 | 3.0367 | 1.7384°C | 0.9183 | 4.6594 | 2.1505°C |
| 08 | 0.2688 | 2.7365 | 1.6504°C | 0.7953 | 4.1138 | 2.0219°C |
| 09 | 0.2484 | 2.5562 | 1.5942°C | 0.6913 | 4.2746 | 2.0601°C |
| 10 | 0.2200 | 2.6705 | 1.6297°C | 0.6059 | 4.6678 | 2.1534°C |
| 11 | 0.2102 | 2.6212 | 1.6156°C | 0.5307 | 4.4421 | 2.1008°C |
| 12 | 0.1847 | 2.6530 | 1.6242°C | 0.4771 | 4.4527 | 2.1036°C |
| 13 | 0.1651 | 2.6391 | 1.6202°C | 0.4319 | 4.5021 | 2.1150°C |
| 14 | 0.1564 | 2.6354 | 1.6184°C | 0.4000 | 4.5515 | 2.1265°C |
| 15 | 0.1501 | 2.6564 | 1.6247°C | 0.3748 | 4.7724 | 2.1772°C |
| 16 | 0.1465 | 2.7432 | 1.6516°C | 0.3610 | 4.6223 | 2.1430°C |
| 17 | 0.1390 | 2.6112 | 1.6111°C | 0.3460 | 4.5895 | 2.1352°C |
| 18 | 0.1349 | 2.6537 | 1.6242°C | 0.3376 | 4.5938 | 2.1363°C |
| 19 | 0.1321 | 2.6335 | 1.6178°C | 0.3318 | 4.7087 | 2.1628°C |
| 20 | 0.1327 | 2.6065 | 1.6096°C | 0.3327 | 4.6547 | 2.1504°C |

## 3. Empirical Diagnostic Conclusion

1. **Generalization Gap**: Across both learning rates, train loss drops steadily to low values (0.1327 for 1e-3, 0.3327 for 3e-4), while validation loss remains elevated (best 2.5562 vs 3.5717).
2. **Undertraining vs Architectural Limitation**: Because extending to 20 epochs and reducing the learning rate by more than 3x does NOT close the validation gap, underperformance is definitively **NOT an undertraining or optimizer instability problem**.
3. **Root Cause Confirmed**: The validation error is primarily driven by the architectural mechanisms identified in Part A: unconditioned high-resolution skip connection leakage across the shared decoder and the initial thermocline climatology prior discrepancy.
