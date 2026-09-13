# SIH26066 — ARGO 2020 Observational Dataset Forensic Audit

**Audit Date**: September 13, 2026  
**Investigator**: Lead Oceanographic Systems & ML Auditor  
**Primary Archive**: Coriolis / INCOIS Global Data Assembly Centre (`https://data-argo.ifremer.fr/geo/indian_ocean/2020/09/`)  
**Status**: **VERIFIED AUTHENTIC & CONTEMPORANEOUS (SEPTEMBER 2020)**  

---

## 1. Executive Summary

This forensic audit verifies the newly acquired contemporaneous ARGO profiling float dataset collected strictly within the North Indian Ocean target domain ($5.0^\circ\text{N} \le \text{lat} \le 30.0^\circ\text{N}, 45.0^\circ\text{E} \le \text{lon} \le 105.0^\circ\text{E}$) during the **September 2020 model test period**.

Unlike the November 2022 dataset used during Phase 4 exploratory benchmarking, this dataset provides **zero-offset contemporaneous in-situ observations** matching the exact calendar dates of the satellite test partition (`chunk_2020_09.pt`).

---

## 2. 10-Point Forensic Inventory

| Audit Item | Forensic Evidence & Verification Result |
| :--- | :--- |
| **1. Observation Date Range** | **2020-09-01T01:07:45 UTC** to **2020-09-25T22:01:26 UTC** (Decoded directly from `JULD`). |
| **2. Total Profiles Acquired** | **37 authentic quality-controlled vertical profiles** inside the NIO basin across 3 snapshot dates (`2020-09-01`, `2020-09-15`, `2020-09-25`). (7 dummy zero-filled profiles from uncalibrated float 2901898 in the 2020-09-25 transmission were forensically detected and excluded). |
| **3. Unique Autonomous Floats** | **28 unique WMO robotic profiling floats** operating simultaneously across the basin. |
| **4. Spatial Coverage** | **Latitude**: $5.6580^\circ\text{N}$ to $20.8061^\circ\text{N}$<br>**Longitude**: $51.0152^\circ\text{E}$ to $92.4460^\circ\text{E}$ |
| **5. Basin Domain Compliance** | **100% strictly compliant**. All 37 profiles fall within $[5^\circ\text{N}, 30^\circ\text{N}] \times [45^\circ\text{E}, 105^\circ\text{E}]$. |
| **6. Pressure / Depth Range** | Sensor pressures span from **$0.20\text{ dbar}$** to **$2,036.0\text{ dbar}$**, with continuous vertical resolution. |
| **7. Temperature Range** | In-situ physical seawater temperatures span from **$2.50^\circ\text{C}$** (at 2000m abyssal depth) to **$29.89^\circ\text{C}$** (surface mixed layer). |
| **8. Missing Values & QC** | All NaN and unphysical values ($\text{TEMP} \le 2.0^\circ\text{C}$ or $\ge 40.0^\circ\text{C}$) excluded. |
| **9. Quality Control Flags** | $>99.2\%$ of sensor readings carry WMO QC flags `1` (Good) or `2` (Probably Good). |
| **10. Target Depth Reaches** | - Reaching $\ge 100\text{ m}$: **37 / 37 profiles** ($100\%$)<br>- Reaching $\ge 200\text{ m}$: **37 / 37 profiles** ($100\%$)<br>- Reaching $\ge 500\text{ m}$: **37 / 37 profiles** ($100\%$)<br>- Reaching $\ge 700\text{ m}$: **36 / 37 profiles** ($97.3\%$)<br>- Reaching $\ge 1000\text{ m}$: **35 / 37 profiles** ($94.6\%$) |

---

## 3. Sub-Basin Spatial Distribution of the 28 Floats

The 37 profiling cycles span all major oceanographic dynamic zones of the North Indian Ocean:
- **Central & Western Arabian Sea**: 14 profiles (capturing the high-salinity Arabian Sea water mass and Somali upwelling filaments).
- **Northern Arabian Sea / Gulf of Oman Approach**: 7 profiles (capturing Persian Gulf outflow water dynamics).
- **Bay of Bengal & Andaman Sea**: 16 profiles (capturing the low-salinity river discharge plume and freshwater stratification).

---

## 4. Machine-Readable Audit Evidence

The detailed profile-by-profile forensic ledger is saved at:
[`reports/argo2020/argo2020_forensic.json`](file:///c:/Users/ankit%20soni/OneDrive/Desktop/SIH/untitled/reports/argo2020/argo2020_forensic.json).
