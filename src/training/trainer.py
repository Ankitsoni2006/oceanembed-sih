"""
SIH26066 — OceanEmbed Production Training Engine
Handles model training loops, validation evaluation, gradient clipping,
checkpoint saving, learning rate scheduling, and depth-stratified metric tracking.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.loss import MaskedMSELoss

logger = logging.getLogger("OceanEmbed.Trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class OceanModelTrainer:
    """
    Production-grade model trainer for OceanEmbed and baseline architectures.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "oceanembed",
        grad_clip_norm: float = 1.0,
        patience: int = 10,
        depth_values: Optional[List[float]] = None
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or MaskedMSELoss()
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.grad_clip_norm = grad_clip_norm
        self.patience = patience
        self.depth_values = depth_values or [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        self.scheduler = scheduler
        self.history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "val_depth_rmse": []
        }
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def _compute_depth_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Computes RMSE per depth layer across valid ocean pixels.
        pred, target: [B, num_depths, H, W]
        """
        metrics = {}
        num_depths = target.shape[1]
        for d in range(num_depths):
            p_d = pred[:, d, :, :]
            t_d = target[:, d, :, :]
            mask = ~torch.isnan(t_d)
            if mask.sum() > 0:
                diff = p_d[mask] - t_d[mask]
                rmse = torch.sqrt(torch.mean(diff ** 2)).item()
                d_label = f"depth_{int(self.depth_values[d])}m" if d < len(self.depth_values) else f"depth_{d}"
                metrics[d_label] = float(rmse)
        return metrics

    def train_epoch(self, epoch: int) -> float:
        """Runs one full training epoch."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        depth_indices = torch.arange(len(self.depth_values), device=self.device)

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(x, depth_indices)
            loss = self.loss_fn(pred, y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Evaluates model on validation loader."""
        if self.val_loader is None:
            return 0.0, {}

        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        depth_indices = torch.arange(len(self.depth_values), device=self.device)
        all_preds = []
        all_targets = []

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x, depth_indices)
            loss = self.loss_fn(pred, y)

            total_loss += loss.item()
            total_batches += 1
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

        avg_loss = total_loss / max(1, total_batches)

        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        depth_rmse = self._compute_depth_metrics(cat_preds, cat_targets)

        return avg_loss, depth_rmse

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Saves model weights and training metadata."""
        ckpt_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_latest.pt")
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "model_name": self.model_name,
            "depth_values": self.depth_values,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(state, ckpt_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
            torch.save(state, best_path)
            logger.info(f"*** New best model saved ({best_path}) with val_loss={val_loss:.4f} ***")

    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Executes full multi-epoch training with early stopping.
        """
        logger.info(f"Starting training '{self.model_name}' for {num_epochs} epochs on device={self.device}...")
        t_start = time.time()

        for epoch in range(1, num_epochs + 1):
            t_epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss, depth_rmse = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.val_loader else train_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)
            self.history["val_depth_rmse"].append(depth_rmse)

            epoch_time = time.time() - t_epoch_start
            logger.info(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Checkpoint & early stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.epochs_without_improvement += 1
                self.save_checkpoint(epoch, val_loss, is_best=False)

            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (patience={self.patience}).")
                break

        total_time = time.time() - t_start
        logger.info(f"Training completed in {total_time:.2f}s. Best val_loss={self.best_val_loss:.4f}")

        # Save history to JSON
        hist_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
