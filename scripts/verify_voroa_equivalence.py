import numpy as np
import torch

from src.models.oceanembed_v3_decoder import OceanEmbedNetV3_Decoder
from src.preprocessing.normalization import OceanStandardScaler
from backend.config import (
    CHECKPOINT_PATH,
    SCALER_PATH,
    TARGET_DEPTHS,
    N_LATS,
    N_LONS,
    LAT_MIN,
    LON_MIN,
    GRID_RESOLUTION,
)


DATE = "2020-09-15"
LAT = 15.0
LON = 85.0


def main():
    print("=" * 70)
    print("OceanEmbed Voroa deployment equivalence test")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------
    device = torch.device("cpu")

    print("\nLoading model...")

    model = OceanEmbedNetV3_Decoder(
        in_vars=7,
        num_depths=len(TARGET_DEPTHS),
        base_features=32,
        embedding_dim=128,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # 2. Load scaler
    # ------------------------------------------------------------
    scaler = OceanStandardScaler.load(str(SCALER_PATH))

    # ------------------------------------------------------------
    # 3. Load original data
    # ------------------------------------------------------------
    old = torch.load(
        "data/processed/chunk_2020_09.pt",
        map_location="cpu",
        weights_only=False,
    )

    dates = old["dates"]

    if DATE not in dates:
        raise RuntimeError(f"{DATE} not found in original chunk")

    offset = dates.index(DATE)

    x_old = old["X"][offset].clone()

    # ------------------------------------------------------------
    # 4. Load deployment data
    # ------------------------------------------------------------
    new = np.load(
        "data/deploy/monthly/2020_09.npz"
    )

    x_new = torch.from_numpy(
        new["X"][offset].astype(np.float32)
    )

    # ------------------------------------------------------------
    # 5. Compare input tensors
    # ------------------------------------------------------------
    input_diff = torch.abs(x_old.float() - x_new)

    print("\nInput comparison")
    print("-" * 70)
    print("Original shape :", tuple(x_old.shape))
    print("Deploy shape   :", tuple(x_new.shape))
    print("Max abs diff   :", float(input_diff.max()))
    print("Mean abs diff  :", float(input_diff.mean()))

    # ------------------------------------------------------------
    # 6. Run original float32 inference
    # ------------------------------------------------------------
    print("\nRunning original float32 inference...")

    with torch.inference_mode():
        old_scaled = scaler.transform(x_old.unsqueeze(0))
        old_pred = model(old_scaled)

    # ------------------------------------------------------------
    # 7. Run deployment-data inference
    # ------------------------------------------------------------
    print("Running deployment float16-storage inference...")

    with torch.inference_mode():
        new_scaled = scaler.transform(x_new.unsqueeze(0))
        new_pred = model(new_scaled)

    # ------------------------------------------------------------
    # 8. Compare complete prediction grids
    # ------------------------------------------------------------
    prediction_diff = torch.abs(old_pred - new_pred)

    print("\nFull-grid prediction comparison")
    print("-" * 70)
    print("Prediction shape:", tuple(old_pred.shape))
    print("Max abs diff    :", float(prediction_diff.max()))
    print("Mean abs diff   :", float(prediction_diff.mean()))
    print("RMSE difference :", float(
        torch.sqrt(torch.mean((old_pred - new_pred) ** 2))
    ))

    # ------------------------------------------------------------
    # 9. Compare exact demo location
    # ------------------------------------------------------------
    lat_idx = int(round((LAT - LAT_MIN) / GRID_RESOLUTION))
    lon_idx = int(round((LON - LON_MIN) / GRID_RESOLUTION))

    old_profile = old_pred[0, :, lat_idx, lon_idx].numpy()
    new_profile = new_pred[0, :, lat_idx, lon_idx].numpy()

    profile_diff = np.abs(old_profile - new_profile)

    print("\nDemo location")
    print("-" * 70)
    print(f"Date       : {DATE}")
    print(f"Location   : {LAT}°N, {LON}°E")
    print(f"Grid index : ({lat_idx}, {lon_idx})")

    print("\nDepth     Original       Deploy        Difference")
    print("-" * 55)

    for depth, old_t, new_t, diff in zip(
        TARGET_DEPTHS,
        old_profile,
        new_profile,
        profile_diff,
    ):
        print(
            f"{depth:>5} m    "
            f"{old_t:>10.6f}    "
            f"{new_t:>10.6f}    "
            f"{diff:>10.6f}"
        )

    print("\n" + "=" * 70)

    print(
        f"Profile max difference: {profile_diff.max():.8f} °C"
    )

    print(
        f"Profile mean difference: {profile_diff.mean():.8f} °C"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()