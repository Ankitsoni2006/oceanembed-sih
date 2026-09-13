"""
SIH26066 — Direct Verification of Copernicus Datasets
Queries Copernicus Marine API directly to verify exact dataset IDs, variables,
temporal ranges, and confirms availability of 2020-01-01 to 2020-01-31.
"""

import json
import copernicusmarine as cm

DATASETS_TO_CHECK = {
    "SST": {
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "variables": ["analysed_sst"]
    },
    "SSS": {
        "dataset_id": "cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c",
        "variables": ["sss"]
    },
    "SSH": {
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "variables": ["sla"]
    },
    "CURRENTS": {
        "dataset_id": "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
        "variables": ["uo", "vo"]
    },
    "WINDS": {
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        "variables": ["eastward_wind", "northward_wind"]
    },
    "GLORYS": {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["thetao"]
    }
}

def verify_dataset(name, cfg):
    did = cfg["dataset_id"]
    print(f"\nVerifying [{name}]: {did}...")
    try:
        desc = cm.describe(dataset_id=did)
        for p in desc.products:
            for d in p.datasets:
                if d.dataset_id == did:
                    for v in d.versions:
                        for part in v.parts:
                            for s in part.services:
                                for c in s.coordinates:
                                    if c.coordinate_id == "time":
                                        print(f"  Service: {s.service_name}")
                                        print(f"  Time Range: {c.minimum_value} to {c.maximum_value}")
                                        return {
                                            "name": name,
                                            "dataset_id": did,
                                            "status": "VERIFIED",
                                            "min_time": str(c.minimum_value),
                                            "max_time": str(c.maximum_value),
                                            "covers_2020_01": (str(c.minimum_value) <= "2020-01-01" and str(c.maximum_value) >= "2020-01-31")
                                        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"name": name, "dataset_id": did, "status": "ERROR", "error": str(e)}

def main():
    report = {}
    for name, cfg in DATASETS_TO_CHECK.items():
        res = verify_dataset(name, cfg)
        report[name] = res

    with open("reports/copernicus_verification.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved verification report to reports/copernicus_verification.json")

if __name__ == "__main__":
    main()
