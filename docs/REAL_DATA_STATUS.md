# SIH26066 OceanEmbed — Real Data & Local File Provenance Status

**Audit Generated:** 2026-09-12  
**Audit Scope:** `data/pilot/`, `data/raw/`, `data/processed/`, `data/argo/`  
**Classification Labels:**
- `VERIFIED`: Authenticated, downloaded directly from Copernicus Marine or Coriolis GDAC via API with validated timestamps and coordinates matching requested dates.
- `PILOT`: Prototype files created during early feasibility testing. May contain date anomalies (e.g. 2026 end-of-catalog dates) or limited test slices.
- `COPIED`: Local file cloned from `data/pilot/` to satisfy preliminary pipeline tests. **Must NOT be counted as historical acquisition.**
- `GENERATED`: Processed PyTorch tensors or NetCDF inference outputs created locally by pipeline code.
- `UNKNOWN`: Origin unverified.

---

## 1. Local Files Detailed Provenance Audit

| Category | Filepath | Size (Bytes) | Temporal Range | Timestamps | Spatial Res | Depths | Status Label | Provenance / Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pilot** | `data/pilot/currents_pilot.nc` | 411,604 | 2020-01-01..2020-01-02 | 2 | 0.25° | 2 (0m, 15m) | `PILOT` | Pre-existing prototype currents file |
| **Pilot** | `data/pilot/glorys_3d_pilot.nc` | 46,904,222 | 2020-01-01..2020-01-03 | 3 | 0.083° | 36 levels | `PILOT` | Pre-existing 3-day GLORYS 3D subset |
| **Pilot** | `data/pilot/glorys_regridded_pilot.nc` | 2,199,602 | **2026-06-23..2026-06-23** | 1 | 0.25° | 1 level | `PILOT` | **Date anomaly**: Contains catalog max date 2026-06-23. Must not be used for 2020 |
| **Pilot** | `data/pilot/glorys_target_15depths_0.25deg.nc` | 8,790,026 | 2020-01-01..2020-01-03 | 3 | 0.25° | 15 levels | `PILOT` | Pre-processed 3-day target file |
| **Pilot** | `data/pilot/sample_X_Y_real.pt` | 2,825,445 | N/A | 1 | 0.25° | 15 levels | `PILOT` | 1-day sanity test tensor |
| **Pilot** | `data/pilot/ssh_pilot.nc` | 796,490 | 2020-01-01..2020-01-02 | 2 | 0.125° | 1 (sfc) | `PILOT` | Pre-existing 2-day SSH pilot |
| **Pilot** | `data/pilot/sss_pilot.nc` | 488,082 | 2019-12-26..2020-01-02 | 2 | 0.20° | 1 (sfc) | `PILOT` | Pre-existing weekly SSS pilot |
| **Pilot** | `data/pilot/sst_regridded_pilot.nc` | 815,470 | **2026-03-31..2026-03-31** | 1 | 0.25° | 1 (sfc) | `PILOT` | **Date anomaly**: Contains catalog max date 2026-03-31. Must not be used for 2020 |
| **Pilot** | `data/pilot/winds_pilot.nc` | 9,244,458 | 2020-01-01..2020-01-01 | 24 (hourly) | 0.125° | 1 (sfc) | `PILOT` | 1-day hourly wind pilot |
| **Raw** | `data/raw/currents/currents_2020-01-01_2020-01-01.nc` | 411,604 | 2020-01-01..2020-01-02 | 2 | 0.25° | 2 | `COPIED` | Copied from `data/pilot/currents_pilot.nc` by early seed script |
| **Raw** | `data/raw/currents/currents_2020-01-01_2020-01-07.nc` | 1,409,908 | 2020-01-01..2020-01-07 | 7 | 0.25° | 2 | `VERIFIED` | Real live Copernicus download (`cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m`) |
| **Raw** | `data/raw/glorys/glorys_2020-01-01_2020-01-01.nc` | 46,904,222 | 2020-01-01..2020-01-03 | 3 | 0.083° | 36 levels | `COPIED` | Copied from `data/pilot/glorys_3d_pilot.nc` by early seed script |
| **Raw** | `data/raw/glorys/glorys_2020-01-01_2020-01-03.nc` | 46,904,222 | 2020-01-01..2020-01-03 | 3 | 0.083° | 36 levels | `COPIED` | Copied from `data/pilot/glorys_3d_pilot.nc` to seed GLORYS cache |
| **Raw** | `data/raw/ssh/ssh_2020-01-01_2020-01-01.nc` | 423,450 | 2020-01-01..2020-01-01 | 1 | 0.125° | 1 (sfc) | `VERIFIED` | Real live Copernicus download (1-day test) |
| **Raw** | `data/raw/ssh/ssh_2020-01-01_2020-01-07.nc` | 2,793,114 | 2020-01-01..2020-01-07 | 7 | 0.125° | 1 (sfc) | `VERIFIED` | Real live Copernicus download (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`) |
| **Raw** | `data/raw/sss/sss_2019-12-24_2020-01-09.nc` | 488,082 | 2019-12-26..2020-01-02 | 2 | 0.20° | 1 (sfc) | `COPIED` | Copied from `data/pilot/sss_pilot.nc` |
| **Raw** | `data/raw/sss/sss_2019-12-25_2020-01-10.nc` | 734,274 | 2019-12-26..2020-01-09 | 3 | 0.20° | 1 (sfc) | `VERIFIED` | Real live Copernicus download (`cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c`) |
| **Raw** | `data/raw/sst/sst_2020-01-01_2020-01-01.nc` | 1,259,732 | 2020-01-01..2020-01-01 | 1 | 0.05° | 1 (sfc) | `VERIFIED` | Real live Copernicus download (1-day test) |
| **Raw** | `data/raw/sst/sst_2020-01-01_2020-01-07.nc` | 8,623,728 | 2020-01-01..2020-01-07 | 7 | 0.05° | 1 (sfc) | `VERIFIED` | Real live Copernicus download (`METOFFICE-GLO-SST-L4-REP-OBS-SST`) |
| **Raw** | `data/raw/winds/test_winds_0125.nc` | 18,985,770 | 2020-01-01..2020-01-02 | 48 (hourly) | 0.125° | 1 (sfc) | `VERIFIED` | Real live Copernicus download (`cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H`) |
| **Raw** | `data/raw/winds/winds_2020-01-01_2020-01-01.nc` | 9,244,458 | 2020-01-01..2020-01-01 | 24 (hourly) | 0.125° | 1 (sfc) | `COPIED` | Copied from `data/pilot/winds_pilot.nc` |
| **Raw** | `data/raw/winds/winds_2020-01-01_2020-01-02.nc` | 18,985,770 | 2020-01-01..2020-01-02 | 48 (hourly) | 0.125° | 1 (sfc) | `VERIFIED` | Exact copy of `test_winds_0125.nc` (real downloaded data) |
| **Processed** | `data/processed/chunk_2020_01_pilot.pt` | 2,826,117 | 2020-01-01 | 1 day | 0.25° | 15 levels | `GENERATED` | Pilot processing run (1 day: 2020-01-01) |
| **Processed** | `data/processed/chunk_2020_01_2day.pt` | 5,649,853 | 2020-01-01..2020-01-02 | 2 days | 0.25° | 15 levels | `GENERATED` | 2-day pipeline test run |
| **Processed** | `data/processed/prediction_output.nc` | 1,767,189 | 2020-01-01 | 1 day | 0.25° | 15 levels | `GENERATED` | Inference output test |
| **Processed** | `data/processed/prediction_output_jan02.nc` | 1,767,189 | 2020-01-02 | 1 day | 0.25° | 15 levels | `GENERATED` | Inference output test |
| **In-Situ** | `data/argo/20221101_prof.nc` | 6,138,940 | 2022-11-01 | Float profiles | Point observations | ARGO PRES | `VERIFIED` | Real Coriolis GDAC in-situ floats (November 2022 snapshot) |

---

## 2. Summary of Copied / Unverified Files to Exclude from Historical Acquisition

1. **`data/raw/glorys/glorys_2020-01-01_2020-01-01.nc` and `glorys_2020-01-01_2020-01-03.nc`**:
   - **Origin**: Copied from `data/pilot/glorys_3d_pilot.nc`.
   - **Action**: These were used ONLY for preliminary code testing. Real GLORYS historical target slices must be downloaded directly via Copernicus Marine API for every target training day.
2. **`data/pilot/sst_regridded_pilot.nc` and `data/pilot/glorys_regridded_pilot.nc`**:
   - **Origin**: Contains anomalous 2026 timestamps.
   - **Action**: Completely excluded from pipeline and acquisition.
3. **`data/raw/sss/sss_2019-12-24_2020-01-09.nc`**:
   - **Origin**: Copied from pilot.
   - **Action**: Real SSS was downloaded as `data/raw/sss/sss_2019-12-25_2020-01-10.nc`. Future multi-day ranges will be fetched directly.
