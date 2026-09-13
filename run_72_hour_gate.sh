#!/bin/bash
set -e

echo "1. Creating Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "2. Installing requirements..."
pip install copernicusmarine xarray netCDF4 numpy pandas

echo "3. Running Pilot Download (Requires Credentials)..."
# Pass credentials as arguments to avoid hardcoding in scripts
read -p "Enter Copernicus Username: " cop_user
read -s -p "Enter Copernicus Password: " cop_pass
echo ""

python3 scripts/01_download_pilot.py --username "$cop_user" --password "$cop_pass"

echo "4. Running Data Verification..."
python3 scripts/02_verify_pilot.py

echo "SUCCESS! You have passed the 72-Hour Data Engineering Gate!"
