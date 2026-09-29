"""
Build deployment-optimized OceanEmbed input data.

Keeps:
    X = 14-channel surface observations + masks

Drops:
    Y = GLORYS target temperatures

Storage:
    float16 + compressed NPZ

The original data/processed/*.pt files are NOT modified.
"""

from pathlib import Path
import json
import shutil
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEPLOY_DIR = PROJECT_ROOT / "data" / "deploy"
MONTHLY_DIR = DEPLOY_DIR / "monthly"


MONTHLY_DIR.mkdir(parents=True, exist_ok=True)


def main():
    manifest = {
        "version": 1,
        "description": "OceanEmbed deployment-only surface input archive",
        "dtype": "float16",
        "channels": 14,
        "shape_per_day": [14, 101, 241],
        "months": {},
        "dates": {},
    }

    total_days = 0

    for month in range(1, 10):
        filename = f"chunk_2020_{month:02d}.pt"
        source = PROCESSED_DIR / filename

        if not source.exists():
            raise FileNotFoundError(f"Missing source chunk: {source}")

        print(f"\nLoading {filename} ...")

        data = torch.load(source, map_location="cpu", weights_only=False)

        if "X" not in data:
            raise KeyError(f"{filename} does not contain X")

        if "dates" not in data:
            raise KeyError(f"{filename} does not contain dates")

        x = data["X"]
        dates = data["dates"]

        print(f"  X shape: {tuple(x.shape)}")
        print(f"  X dtype: {x.dtype}")
        print(f"  Days:    {len(dates)}")

        # Convert only X to NumPy float16.
        #
        # The model will convert the selected daily field back to float32
        # before normalization/inference.
        x_np = x.detach().cpu().numpy().astype(np.float16, copy=False)

        output_name = f"2020_{month:02d}.npz"
        output_path = MONTHLY_DIR / output_name

        print(f"  Writing {output_name} ...")

        np.savez_compressed(
            output_path,
            X=x_np,
        )

        manifest["months"][f"2020-{month:02d}"] = {
            "file": output_name,
            "days": len(dates),
            "shape": list(x_np.shape),
            "dtype": "float16",
        }

        for offset, date in enumerate(dates):
            manifest["dates"][date] = {
                "month": f"2020-{month:02d}",
                "file": output_name,
                "offset": offset,
            }

        total_days += len(dates)

        del data
        del x
        del x_np

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Compressed size: {size_mb:.2f} MB")

    manifest["total_days"] = total_days

    manifest_path = DEPLOY_DIR / "dates.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("Deployment data build complete")
    print("=" * 60)

    total_size = 0

    for path in sorted(MONTHLY_DIR.glob("*.npz")):
        size_mb = path.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"{path.name:15s} {size_mb:8.2f} MB")

    print("-" * 60)
    print(f"Total compressed X data: {total_size:.2f} MB")
    print(f"Total dates:             {total_days}")
    print(f"Manifest:                {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()