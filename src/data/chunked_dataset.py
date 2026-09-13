"""
SIH26066 — OceanEmbed Chunked Dataset & Storage Engine
Provides out-of-core chunked storage, lazy loading, and dataset index mapping
for multi-year scalability without memory bottlenecks.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.normalization import OceanStandardScaler

logger = logging.getLogger("OceanEmbed.Storage")


class ChunkedOceanDataset(Dataset):
    """
    Multi-year out-of-core PyTorch Dataset.
    Loads and caches monthly tensor chunks dynamically on access.
    Memory footprint remains strictly bounded regardless of historical span.
    """
    def __init__(
        self,
        processed_dir: str = "data/processed",
        scaler: Optional[OceanStandardScaler] = None,
        dates: Optional[List[str]] = None
    ):
        self.processed_dir = processed_dir
        self.scaler = scaler
        self.index_manifest_path = os.path.join(self.processed_dir, "dataset_index.json")
        
        self.samples: List[Dict[str, Any]] = []
        self._cached_chunk_id: Optional[str] = None
        self._cached_x: Optional[torch.Tensor] = None
        self._cached_y: Optional[torch.Tensor] = None
        
        if os.path.exists(self.index_manifest_path):
            with open(self.index_manifest_path, "r") as f:
                manifest_data = json.load(f)
                all_samples = manifest_data.get("samples", [])
                if dates is not None:
                    dates_set = set(dates)
                    self.samples = [s for s in all_samples if s["date"] in dates_set]
                else:
                    self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_chunk_to_cache(self, chunk_file: str):
        chunk_path = os.path.join(self.processed_dir, chunk_file)
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        self._cached_chunk_id = chunk_file
        self._cached_x = data["X"]
        self._cached_y = data["Y"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            x: [14, 101, 241] normalized input tensor
            y: [15, 101, 241] target temperature profile tensor
            date: string YYYY-MM-DD
        """
        meta = self.samples[idx]
        chunk_file = meta["chunk_file"]
        offset = meta["offset"]
        date_str = meta["date"]

        if self._cached_chunk_id != chunk_file:
            self._load_chunk_to_cache(chunk_file)

        x = self._cached_x[offset]  # [14, 101, 241]
        y = self._cached_y[offset]  # [15, 101, 241]

        # Apply normalization if scaler provided
        if self.scaler is not None:
            x = self.scaler.transform(x)

        return x, y, date_str

    @classmethod
    def save_chunk(
        cls,
        processed_dir: str,
        chunk_name: str,
        x_list: List[torch.Tensor],
        y_list: List[torch.Tensor],
        dates: List[str],
        metadata_extra: Optional[dict] = None
    ) -> str:
        """
        Saves a batch of daily samples into a standardized chunk file and updates dataset_index.json.
        """
        os.makedirs(processed_dir, exist_ok=True)
        chunk_filename = f"{chunk_name}.pt"
        chunk_path = os.path.join(processed_dir, chunk_filename)

        # Concatenate along time/batch dim -> [T, 14, 101, 241] and [T, 15, 101, 241]
        X_chunk = torch.cat(x_list, dim=0) if x_list[0].ndim == 4 else torch.stack(x_list, dim=0)
        Y_chunk = torch.cat(y_list, dim=0) if y_list[0].ndim == 4 else torch.stack(y_list, dim=0)

        payload = {
            "chunk_name": chunk_name,
            "dates": dates,
            "X": X_chunk,
            "Y": Y_chunk,
            "metadata": metadata_extra or {}
        }
        torch.save(payload, chunk_path)
        logger.info(f"Saved chunk '{chunk_name}' to {chunk_path} ({os.path.getsize(chunk_path):,} bytes, {len(dates)} days)")

        # Update dataset index manifest
        index_file = os.path.join(processed_dir, "dataset_index.json")
        existing_index = {"samples": [], "chunks": {}}
        if os.path.exists(index_file):
            try:
                with open(index_file, "r") as f:
                    existing_index = json.load(f)
            except Exception:
                pass

        # Remove previous entries for this chunk if re-saving
        existing_index["samples"] = [s for s in existing_index["samples"] if s["chunk_file"] != chunk_filename]
        
        for offset, date_str in enumerate(dates):
            existing_index["samples"].append({
                "date": date_str,
                "chunk_file": chunk_filename,
                "offset": offset
            })

        existing_index["chunks"][chunk_name] = {
            "file": chunk_filename,
            "num_days": len(dates),
            "date_start": dates[0],
            "date_end": dates[-1],
            "size_bytes": os.path.getsize(chunk_path)
        }

        # Sort samples chronologically
        existing_index["samples"].sort(key=lambda s: s["date"])

        with open(index_file, "w") as f:
            json.dump(existing_index, f, indent=2)

        return chunk_path
