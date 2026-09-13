import unittest
import torch
from src.preprocessing.masks import generate_valid_mask, concatenate_variables_and_masks
from src.training.loss import MaskedMSELoss
from src.models.oceanembed import OceanEmbedNet

class TestDataLeakageAndPipeline(unittest.TestCase):
    def test_temporal_overlap(self):
        train_years = set(range(2010, 2020))
        val_years = set(range(2020, 2022))
        test_years = set(range(2022, 2024))
        
        self.assertTrue(len(train_years.intersection(val_years)) == 0, "Temporal leakage between train and val")
        self.assertTrue(len(train_years.intersection(test_years)) == 0, "Temporal leakage between train and test")
        self.assertTrue(len(val_years.intersection(test_years)) == 0, "Temporal leakage between val and test")

    def test_mask_generation(self):
        test_tensor = torch.tensor([
            [1.0, 0.0, -999.0],
            [float('nan'), 2.0, 3.0]
        ])
        mask = generate_valid_mask(test_tensor, missing_val_threshold=-100.0)
        expected_mask = torch.tensor([
            [1., 1., 0.],
            [0., 1., 1.]
        ])
        self.assertTrue(torch.equal(mask, expected_mask), "Mask generation failed")

    def test_tensor_concatenation(self):
        test_tensor = torch.randn(2, 7, 10, 10)
        test_tensor[0, 0, 5, 5] = float('nan')
        result = concatenate_variables_and_masks(test_tensor, -100.0)
        self.assertEqual(result.shape, (2, 14, 10, 10), "Channel concatenation shape is incorrect")
        self.assertFalse(torch.isnan(result[:, 0:7, :, :]).any(), "NaNs leaked into safe variables tensor")
        self.assertEqual(result[0, 7, 5, 5].item(), 0.0, "Mask failed to record missing value")

    def test_masked_mse_nan_safety(self):
        pred = torch.randn(2, 15, 10, 10, requires_grad=True)
        target = torch.randn(2, 15, 10, 10)
        target[:, :, :5, :] = float('nan') # Half the target is land/NaNs
        criterion = MaskedMSELoss()
        loss = criterion(pred, target)
        self.assertFalse(torch.isnan(loss), "MaskedMSELoss produced NaN with land pixels")
        loss.backward()
        self.assertIsNotNone(pred.grad, "Grad should exist")
        self.assertFalse(torch.isnan(pred.grad).any(), "Grad contained NaNs")

    def test_oceanembed_batch_scaling(self):
        model = OceanEmbedNet(in_vars=7, num_depths=15, base_features=8, embedding_dim=16)
        depth_indices = torch.arange(15)
        for B in [1, 2, 4]:
            x = torch.randn(B, 14, 101, 241)
            out = model(x, depth_indices)
            self.assertEqual(out.shape, (B, 15, 101, 241), f"OceanEmbed failed shape test for batch size {B}")

if __name__ == '__main__':
    unittest.main()
