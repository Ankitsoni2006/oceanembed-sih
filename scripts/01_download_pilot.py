import os
import subprocess
import argparse

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"ERROR:\n{result.stderr}")
    else:
        print(f"SUCCESS:\n{result.stdout}")

def main():
    parser = argparse.ArgumentParser(description="Download Pilot Data for SIH26066")
    parser.add_argument("--username", required=True, help="Copernicus Username")
    parser.add_argument("--password", required=True, help="Copernicus Password")
    args = parser.parse_args()

    # Define the region and pilot time (1 day)
    lon_min, lon_max = 45, 105
    lat_min, lat_max = 5, 30
    date_start = "2022-11-12"
    date_end = "2022-11-12"
    out_dir = "data/pilot"
    
    os.makedirs(out_dir, exist_ok=True)

    print("Authenticating...")
    # NOTE: Never hardcode passwords in scripts!
    
    # 1. Download GLORYS 3D Target (thetao)
    glorys_cmd = f"""
    copernicusmarine subset \\
      --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \\
      --variable thetao \\
      --start-datetime {date_start} --end-datetime {date_end} \\
      --minimum-longitude {lon_min} --maximum-longitude {lon_max} \\
      --minimum-latitude {lat_min} --maximum-latitude {lat_max} \\
      --minimum-depth 0 --maximum-depth 1000 \\
      --username {args.username} --password '{args.password}' \\
      --output-directory {out_dir} \\
      --output-filename glorys_pilot.nc \\
      --force-download
    """
    
    # 2. Download SST Input (OSTIA)
    sst_cmd = f"""
    copernicusmarine subset \\
      --dataset-id cmems_obs-sst_glo_phy_my_l4_P1D-m \\
      --variable analysed_sst \\
      --start-datetime {date_start} --end-datetime {date_end} \\
      --minimum-longitude {lon_min} --maximum-longitude {lon_max} \\
      --minimum-latitude {lat_min} --maximum-latitude {lat_max} \\
      --username {args.username} --password '{args.password}' \\
      --output-directory {out_dir} \\
      --output-filename sst_pilot.nc \\
      --force-download
    """

    print("=== Downloading GLORYS 3D PILOT ===")
    run_cmd(glorys_cmd)
    
    print("=== Downloading SST PILOT ===")
    run_cmd(sst_cmd)

if __name__ == "__main__":
    main()
