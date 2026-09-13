"""
SIH26066 — OceanEmbed End-to-End System Integration Test Suite
Tests every major milestone: Catalog, Preprocessing, Normalization, Models,
Training, Spatial Evaluation, ARGO Colocation, Inference, and FastAPI REST API.
"""

import os
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.data.catalog import DATA_CATALOG, TARGET_GRID, TARGET_DEPTHS, COMMON_USABLE_PERIOD
from src.preprocessing.normalization import OceanStandardScaler
from src.data.chunked_dataset import ChunkedOceanDataset
from src.models.oceanembed import OceanEmbedNet
from src.models.baselines import PointwiseMLP, SimpleCNNBaseline
from src.evaluation.metrics import calculate_metrics
from src.inference.predict import OceanEmbedPredictor
from api.server import app


class TestOceanEmbedE2E(unittest.TestCase):
    def test_01_catalog_integrity(self):
        """Verifies North Indian Ocean grid, 15 depths, and 6 catalog specs."""
        self.assertEqual(TARGET_GRID.shape, (101, 241))
        self.assertEqual(TARGET_DEPTHS.num_depths, 15)
        self.assertEqual(list(TARGET_DEPTHS.depths), [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])
        for required_key in ["SST", "SSS", "SSH", "CURRENTS", "WINDS", "GLORYS"]:
            self.assertIn(required_key, DATA_CATALOG)
            spec = DATA_CATALOG[required_key]
            self.assertTrue(len(spec.variables) > 0)
            self.assertTrue(len(spec.dataset_id) > 0)

    def test_02_normalization_cycle(self):
        """Verifies fit, transform, inverse transform, and JSON persistence."""
        scaler = OceanStandardScaler()
        test_x = np.random.randn(2, 14, 101, 241).astype(np.float32)
        # Masks set to 1
        test_x[:, 7:14, :, :] = 1.0
        scaler.fit(test_x)
        self.assertEqual(len(scaler.means), 7)
        self.assertEqual(len(scaler.stds), 7)

        scaled = scaler.transform(test_x)
        self.assertEqual(scaled.shape, test_x.shape)
        # Masks should remain exactly 1.0
        self.assertTrue(np.allclose(scaled[:, 7:14, :, :], 1.0))

        # Test export / import
        export_path = "checkpoints/test_scaler.json"
        scaler.save(export_path)
        self.assertTrue(os.path.exists(export_path))
        scaler2 = OceanStandardScaler.load(export_path)
        self.assertTrue(np.allclose(scaler.means, scaler2.means))
        if os.path.exists(export_path):
            os.remove(export_path)

    def test_03_processed_dataset_chunks(self):
        """Verifies chunk existence and PyTorch DataLoader loading."""
        chunk_path = "data/processed/chunk_2020_01_2day.pt"
        self.assertTrue(os.path.exists(chunk_path), f"Processed chunk missing at {chunk_path}")
        ds = ChunkedOceanDataset(processed_dir="data/processed")
        self.assertGreaterEqual(len(ds), 2)
        x0, y0, date0 = ds[0]
        self.assertEqual(x0.shape, (14, 101, 241))
        self.assertEqual(y0.shape, (15, 101, 241))
        self.assertTrue(len(date0) == 10)

    def test_04_model_architectures_forward(self):
        """Verifies forward pass for MLP, SimpleCNN, and OceanEmbedNet."""
        x = torch.randn(1, 14, 101, 241)
        depth_indices = torch.arange(15)

        mlp = PointwiseMLP()
        cnn = SimpleCNNBaseline()
        dl = OceanEmbedNet()

        out_mlp = mlp(x)
        out_cnn = cnn(x)
        out_dl = dl(x, depth_indices)

        self.assertEqual(out_mlp.shape, (1, 15, 101, 241))
        self.assertEqual(out_cnn.shape, (1, 15, 101, 241))
        self.assertEqual(out_dl.shape, (1, 15, 101, 241))

    def test_05_scientific_metrics_calculation(self):
        """Verifies RMSE, MAE, Bias, and Pearson r on synthetic tensors."""
        pred = torch.tensor([[10.0, 20.0], [30.0, float("nan")]])
        target = torch.tensor([[11.0, 21.0], [31.0, float("nan")]])

        m = calculate_metrics(pred, target)
        self.assertAlmostEqual(m["mae"], 1.0, places=3)
        self.assertAlmostEqual(m["rmse"], 1.0, places=3)
        self.assertAlmostEqual(m["bias"], -1.0, places=3)
        self.assertAlmostEqual(m["corr"], 1.0, places=3)

    def test_06_inference_and_physical_indices(self):
        """Verifies MLD, Thermocline depth, and OHC computation."""
        predictor = OceanEmbedPredictor()
        # Synthetic column: warm surface, sharp drop, cold deep
        temp_3d = np.zeros((15, 101, 241), dtype=np.float32)
        profile = [28.0, 28.0, 27.9, 27.5, 27.0, 24.0, 20.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0]
        for d_i, t_val in enumerate(profile):
            temp_3d[d_i, :, :] = t_val

        indices = predictor.compute_oceanographic_indices(temp_3d)
        self.assertIn("mld_meters", indices)
        self.assertIn("thermocline_depth_meters", indices)
        self.assertIn("ohc300_gj_m2", indices)
        self.assertTrue(indices["mld_meters"][50, 50] > 0)
        self.assertTrue(indices["ohc300_gj_m2"][50, 50] > 0)

    def test_07_api_server_endpoints(self):
        """Verifies all FastAPI routes."""
        client = TestClient(app)
        
        # /health
        res = client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "HEALTHY")

        # /catalog
        res = client.get("/catalog")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["input_channels"]), 14)

        # /predict
        res = client.post("/predict", json={"lat": 15.0, "lon": 65.0, "date": "2020-01-01"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["profile"]), 15)

        # /evaluation
        res = client.get("/evaluation")
        self.assertEqual(res.status_code, 200)
        self.assertIn("overall_metrics", res.json())

        # /argo-validation
        res = client.get("/argo-validation")
        self.assertEqual(res.status_code, 200)
        self.assertIn("overall_metrics", res.json())

        # /training-history
        res = client.get("/training-history?model=oceanembed")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(len(res.json()["epoch"]) > 0)


if __name__ == "__main__":
    unittest.main()
