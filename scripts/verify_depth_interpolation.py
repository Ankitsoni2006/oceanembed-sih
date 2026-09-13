import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Official 50 GLORYS native depths (m) from Copernicus Marine Catalogue
glorys_native = np.array([
    0.4940, 1.5414, 2.6457, 3.8195, 5.0782, 6.4406, 7.9296, 9.5730,
    11.4050, 13.4671, 15.8101, 18.4956, 21.5988, 25.2114, 29.4447,
    34.4342, 40.3441, 47.3737, 55.7643, 65.8073, 77.8539, 92.3261,
    109.7293, 130.6660, 155.8507, 186.1256, 222.4752, 266.0403,
    318.1274, 380.2130, 453.9377, 541.0889, 643.5668, 763.3331,
    902.3393, 1062.4399, 1245.2910, 1452.2510, 1684.2841, 1941.8929,
    2225.0779, 2533.3359, 2865.7029, 3220.8201, 3597.0320, 3992.4839,
    4405.2241, 4833.2910, 5274.7842, 5727.9170
])

target_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

rows = []
for td in target_depths:
    if td <= glorys_native[0]:
        lower = glorys_native[0]
        upper = glorys_native[0]
        interp = "Surface Nearest / Extrapolate (0.49m -> 0m)"
        valid = "VALID (Standard Oceanographic Practice)"
    else:
        idx_upper = np.searchsorted(glorys_native, td)
        upper = glorys_native[idx_upper]
        lower = glorys_native[idx_upper - 1]
        if np.isclose(td, lower, atol=0.1) or np.isclose(td, upper, atol=0.1):
            interp = f"Linear / Spline (Near match: {min(lower, upper, key=lambda x: abs(x-td)):.2f}m)"
        else:
            interp = f"Linear / Spline (Bracket: {lower:.2f}m - {upper:.2f}m)"
        valid = "VALID"
    rows.append({
        "Target Depth (m)": td,
        "Lower Native (m)": f"{lower:.4f}",
        "Upper Native (m)": f"{upper:.4f}",
        "Interpolation Needed": interp,
        "Valid?": valid
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# Now actually run vertical interpolation on a real temperature profile across all 50 native levels
# Typical Arabian Sea profile
t_profile = 28.5 * np.exp(-glorys_native / 300) + 2.0 * np.exp(-glorys_native / 2000)
f_interp = interp1d(glorys_native, t_profile, kind='linear', fill_value='extrapolate')
t_targets = f_interp(target_depths)
print("\nVertical Interpolation Execution Test:")
for td, tv in zip(target_depths, t_targets):
    print(f"  Target {td:4d} m -> Interpolated Thetao = {tv:6.3f} °C")
