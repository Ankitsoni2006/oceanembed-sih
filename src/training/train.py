"""
SIH26066 — OceanEmbed Model Training Entry Point
Configures and launches training runs for OceanEmbedNet and baseline architectures
with strict chronological train/validation splitting and comprehensive metric logging.
"""

import os
import sys
import json
import argparse
import logging
from typing import Tuple, Optional, List, Dict
import torch
from torch.utils.data import DataLoader, Subset

from src.models.oceanembed import OceanEmbedNet
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.training.trainer import OceanModelTrainer
from src.data.chunked_dataset import ChunkedOceanDataset
from src.data.catalog import TARGET_DEPTHS

logger = logging.getLogger("OceanEmbed.Train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_model(model_type: str, num_depths: int = 15) -> torch.nn.Module:
    """Instantiates the requested architecture."""
    model_type = model_type.lower()
    if model_type == "oceanembed":
        logger.info("Instantiating OceanEmbedNet (Multi-scale U-Net Encoder + Latent Bottleneck + Depth-Conditioned Decoder)...")
        return OceanEmbedNet(in_vars=7, num_depths=num_depths, base_features=32, embedding_dim=128)
    elif model_type == "mlp":
        logger.info("Instantiating Baseline PointwiseMLP (Independent spatial pixel regression)...")
        return PointwiseMLP(in_vars=7, num_depths=num_depths, hidden_dim=64)
    elif model_type in ["simple_cnn", "cnn"]:
        logger.info("Instantiating Baseline SimpleCNNBaseline (Spatial CNN without skip connections or latent bottleneck)...")
        return SimpleCNNBaseline(in_vars=7, num_depths=num_depths, hidden_dim=64)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: oceanembed, mlp, simple_cnn")


from src.preprocessing.normalization import OceanStandardScaler

def create_chronological_loaders(
    dataset_path: str,
    val_ratio: float = 0.2,
    batch_size: int = 1,
    num_workers: int = 0,
    scaler_path: Optional[str] = "configs/scaler_params_real.json"
) -> Tuple[DataLoader, DataLoader, list, OceanStandardScaler]:
    """
    Creates DataLoaders with strict chronological (temporal) splitting.
    Fits or loads OceanStandardScaler strictly on the training partition.
    Random shuffling across time is strictly prohibited in ocean forecasting.
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    raw_pt = torch.load(dataset_path, weights_only=False)
    x = raw_pt["X"]
    y = raw_pt["Y"]
    dates = raw_pt.get("dates", [])
    num_samples = x.shape[0]

    logger.info(f"Dataset contains {num_samples} total day samples ({dates[0] if dates else 'N/A'} to {dates[-1] if dates else 'N/A'})")

    if num_samples == 1:
        logger.warning("Dataset contains only 1 sample. Using the single sample for both train and validation sanity check.")
        train_indices = [0]
        val_indices = [0]
    else:
        num_val = max(1, int(round(num_samples * val_ratio)))
        num_train = num_samples - num_val
        train_indices = list(range(0, num_train))
        val_indices = list(range(num_train, num_samples))

    logger.info(f"Train split: {len(train_indices)} samples ({[dates[i] for i in train_indices if i < len(dates)]})")
    logger.info(f"Val split:   {len(val_indices)} samples ({[dates[i] for i in val_indices if i < len(dates)]})")

    # Fit or load standard scaler strictly on training split
    x_train_raw = x[train_indices]
    x_val_raw = x[val_indices]

    if scaler_path and os.path.exists(scaler_path):
        logger.info(f"Loading fitted scaler from {scaler_path}")
        scaler = OceanStandardScaler.load(scaler_path)
    else:
        logger.info("Fitting OceanStandardScaler strictly on training partition...")
        scaler = OceanStandardScaler(num_physical_channels=7)
        scaler.fit(x_train_raw)
        if scaler_path:
            scaler.save(scaler_path)
            logger.info(f"Saved scaler parameters to {scaler_path}")

    x_train_norm = scaler.transform(x_train_raw)
    x_val_norm = scaler.transform(x_val_raw)

    # Construct custom in-memory tensor dataset for fast access
    class TensorOceanDataset(torch.utils.data.Dataset):
        def __init__(self, x_tensor, y_tensor):
            self.x = x_tensor
            self.y = y_tensor
        def __len__(self):
            return self.x.shape[0]
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    train_ds = TensorOceanDataset(x_train_norm, y[train_indices])
    val_ds = TensorOceanDataset(x_val_norm, y[val_indices])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, dates, scaler


def run_training(
    model_type: str = "oceanembed",
    dataset_path: str = "data/processed/chunk_2020_01_pilot.pt",
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 1,
    val_ratio: float = 0.2,
    patience: int = 10,
    checkpoint_dir: str = "checkpoints",
    scaler_path: Optional[str] = None
):
    if scaler_path is None:
        if "month" in dataset_path or "31day" in dataset_path:
            scaler_path = "configs/scaler_params_real_month.json"
        else:
            scaler_path = "configs/scaler_params_real.json"

    train_loader, val_loader, dates, scaler = create_chronological_loaders(
        dataset_path=dataset_path,
        val_ratio=val_ratio,
        batch_size=batch_size,
        scaler_path=scaler_path
    )

    model = build_model(model_type, num_depths=TARGET_DEPTHS.num_depths)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model '{model_type}' initialized with {num_params:,} trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    trainer = OceanModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        model_name=model_type,
        patience=patience,
        depth_values=list(TARGET_DEPTHS.depths)
    )

    history = trainer.train(num_epochs=epochs)
    return history


def main():
    parser = argparse.ArgumentParser(description="SIH26066 — OceanEmbed Model Training")
    parser.add_argument("--model", default="oceanembed", choices=["oceanembed", "mlp", "simple_cnn"], help="Model architecture to train")
    parser.add_argument("--dataset", default="data/processed/chunk_2020_01_pilot.pt", help="Path to processed .pt dataset chunk")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--scaler-path", default=None, help="Path to precomputed scaler params JSON")

    args = parser.parse_args()
    run_training(
        model_type=args.model,
        dataset_path=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        scaler_path=args.scaler_path
    )


if __name__ == "__main__":
    main()
