import requests
import re

url = "https://data-argo.ifremer.fr/geo/indian_ocean/2020/09/"
headers = {"User-Agent": "Mozilla/5.0"}
print(f"Checking URL: {url}")
try:
    r = requests.get(url, headers=headers, timeout=25)
    print(f"Status code: {r.status_code}")
    files = re.findall(r'href="([^"]+\.nc)"', r.text)
    print(f"Total NetCDF files found: {len(files)}")
    if files:
        print("Sample files:", files[:10])
except Exception as e:
    print(f"Error accessing GDAC: {e}")
