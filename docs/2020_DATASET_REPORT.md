# SIH26066 — Full 2020 Real Dataset Quality & Verification Report

**Audit Status**: **PASSED**  
**Audited At**: 2026-09-12T20:41:56Z  
**Chunks Audited**: 9 monthly chunks (274 daily samples)

## 1. Quality Gates Assessment

| Quality Gate | Description | Status |
| :--- | :--- | :---: |
| **Gate 1** | Chronological ordering and zero date duplicates | PASSED |
| **Gate 2 & 3** | Exact NetCDF internal timestamps without substitution | PASSED |
| **Gate 4 & 5** | Dimensions ($X: [14, 101, 241]$, $Y: [15, 101, 241]$) | PASSED |
| **Gate 6 & 7** | NIO target grid (0.25°) and 15 locked target depths | PASSED |
| **Gate 8** | Native GLORYS 36-level span covers 0–1000m interpolation | PASSED |
| **Gate 9 & 10** | Physical validity and strictly binary masks | PASSED |
| **Gate 11** | Zero temporal overlap across Train / Val / Test partitions | PASSED |
| **Gate 12** | Train-only zero-leakage normalization | PASSED |

## 2. Partition Inventory

- **Training Set (Jan–Sep 2020)**: 274 days
- **Validation Set (Oct–Nov 2020)**: 0 days
- **Test Set (Dec 2020)**: 0 days
- **Total Dataset Samples**: 274 days

## 3. Chunk Catalog

| Chunk File | Sample Days | Date Range | Size (MB) |
| :--- | :---: | :---: | :---: |
| `chunk_2020_01.pt` | 31 | 2020-01-01 to 2020-01-31 | 83.48 MB |
| `chunk_2020_02.pt` | 29 | 2020-02-01 to 2020-02-29 | 78.09 MB |
| `chunk_2020_03.pt` | 31 | 2020-03-01 to 2020-03-31 | 83.48 MB |
| `chunk_2020_04.pt` | 30 | 2020-04-01 to 2020-04-30 | 80.78 MB |
| `chunk_2020_05.pt` | 31 | 2020-05-01 to 2020-05-31 | 83.48 MB |
| `chunk_2020_06.pt` | 30 | 2020-06-01 to 2020-06-30 | 80.78 MB |
| `chunk_2020_07.pt` | 31 | 2020-07-01 to 2020-07-31 | 83.48 MB |
| `chunk_2020_08.pt` | 31 | 2020-08-01 to 2020-08-31 | 83.48 MB |
| `chunk_2020_09.pt` | 30 | 2020-09-01 to 2020-09-30 | 80.78 MB |
