import xarray as xr
from datetime import datetime

# Check the downloaded files and verified catalogue bounds
time_ranges = {
    "GLORYS": ("1993-01-01", "2026-06-23"),
    "SST (OSTIA)": ("1981-08-24", "2026-03-31"),
    "SSS (Multi-Obs)": ("2010-06-03", "2025-12-25"),
    "SSH (DUACS)": ("1993-01-01", "2026-03-31"),
    "U Current (Multi-Obs)": ("1993-01-01", "2026-03-31"),
    "V Current (Multi-Obs)": ("1993-01-01", "2026-03-31"),
    "U Wind (0.125° L4)": ("2007-01-11", "2026-04-21"),
    "V Wind (0.125° L4)": ("2007-01-11", "2026-04-21")
}

print("============================================================")
print("COMMON TIME PERIOD DETERMINATION")
print("============================================================")

earliest_dates = []
latest_dates = []

for name, (start, end) in time_ranges.items():
    d_s = datetime.strptime(start, "%Y-%m-%d")
    d_e = datetime.strptime(end, "%Y-%m-%d")
    earliest_dates.append((d_s, name))
    latest_dates.append((d_e, name))
    print(f" - {name:22s} : {start} to {end}")

# Earliest common date is the maximum of all start dates
earliest_common, limiting_start_dataset = max(earliest_dates, key=lambda x: x[0])
# Latest common date is the minimum of all end dates
latest_common, limiting_end_dataset = min(latest_dates, key=lambda x: x[0])

print("\n--- RESULTS ---")
print(f"Earliest Common Date: {earliest_common.strftime('%Y-%m-%d')} (Limiting dataset: {limiting_start_dataset})")
print(f"Latest Common Date:   {latest_common.strftime('%Y-%m-%d')} (Limiting dataset: {limiting_end_dataset})")
print(f"Total overlapping span: {(latest_common - earliest_common).days} days ({((latest_common - earliest_common).days)/365.25:.1f} years)")
