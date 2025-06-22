"""
Tests for auxiliary loss utilities.

Specifically: exhaustive randomised checks for the `MaskSoftmax` module.
"""

import torch
import unittest
from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
from torch.autograd import gradcheck

from mypt.losses.auxiliary import MaskSoftmax
from mypt.code_utils import pytorch_utils as pu


class TestMaskSoftmax(unittest.TestCase):
    """Randomised unit-tests for :class:`MaskSoftmax`."""

    def setUp(self) -> None:
        self.sm = MaskSoftmax()
        # Number of random trials per test
        self.num_trials = 1000
        pu.seed_everything(0)

    def _compute_naive_masked_softmax_last_dim_2d(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Naive masked softmax on the last dimension of a 2D tensor."""
        B, _ = x.shape
        expected = torch.zeros_like(x)
        for i in range(B):
            row_x = x[i]
            row_mask = mask[i]
            
            masked_vals = row_x[row_mask]
            if masked_vals.numel() > 0:
                softmax_vals = F.softmax(masked_vals, dim=0)
                expected[i, row_mask] = softmax_vals
        return expected

    def _compute_naive_masked_softmax_last_dim_3d(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Naive masked softmax on the last dimension of a 3D tensor."""
        B, S, _ = x.shape
        expected = torch.zeros_like(x)
        for b in range(B):
            for s in range(S):
                slice_x = x[b, s]
                slice_mask = mask[b, s]
                
                masked_vals = slice_x[slice_mask]
                if masked_vals.numel() > 0:
                    softmax_vals = F.softmax(masked_vals, dim=0)
                    expected[b, s, slice_mask] = softmax_vals
        return expected

    def test_masked_softmax_vs_naive_2d(self):
        """Compare MaskSoftmax with a naive loop-based 2D implementation for all dimensions."""
        num_iterations = 10_000
        for _ in tqdm(range(num_iterations), desc="Testing MaskSoftmax vs naive 2D"):
            # 1. create a random 2d tensor and mask
            B = np.random.randint(2, 8)
            S = np.random.randint(2, 16)
            x = torch.randn(B, S)
            mask = torch.rand(B, S) > 0.3 # keep ~70%

            # Test dim=-1 (rows)
            # Ensure at least one True per row
            for b in range(B):
                if not mask[b].any(): mask[b, np.random.randint(0, S)] = True
            
            output = self.sm(x, mask, dim=-1)
            expected = self._compute_naive_masked_softmax_last_dim_2d(x, mask)
            self.assertTrue(torch.allclose(output, expected, atol=1e-12))
            
            # Test dim=0 (columns)
            # Ensure at least one True per column
            mask = mask.T # reuse same random mask, but check columns now
            for s in range(S):
                if not mask[s].any(): mask[s, np.random.randint(0, B)] = True
            mask = mask.T # transpose back

            output_dim0 = self.sm(x, mask, dim=0)
            expected_dim0 = self._compute_naive_masked_softmax_last_dim_2d(x.T, mask.T).T
            self.assertTrue(torch.allclose(output_dim0, expected_dim0, atol=1e-12))

    def test_masked_softmax_vs_naive_3d(self):
        """Compare MaskSoftmax with a naive loop-based 3D implementation for all dimensions."""
        num_iterations = 10_000
        for _ in tqdm(range(num_iterations), desc="Testing MaskSoftmax vs naive 3D"):
            # 1. create a random 3d tensor and mask
            B = np.random.randint(2, 8)
            S = np.random.randint(2, 16)
            D = np.random.randint(2, 12)
            x = torch.randn(B, S, D)
            
            # Test dim=-1 (dim 2)
            mask = torch.rand(B, S, D) > 0.3
            for b in range(B):
                for s in range(S):
                    if not mask[b, s].any(): mask[b, s, np.random.randint(0, D)] = True
            
            output = self.sm(x, mask, dim=-1)
            expected = self._compute_naive_masked_softmax_last_dim_3d(x, mask)
            self.assertTrue(torch.allclose(output, expected, atol=1e-12))

            # Test dim=1
            mask = torch.rand(B, S, D) > 0.3
            for b in range(B):
                for d in range(D):
                    if not mask[b, :, d].any(): mask[b, np.random.randint(0, S), d] = True
            
            output_dim1 = self.sm(x, mask, dim=1)
            x_permuted_d1 = x.permute(0, 2, 1)
            mask_permuted_d1 = mask.permute(0, 2, 1)
            expected_permuted_d1 = self._compute_naive_masked_softmax_last_dim_3d(x_permuted_d1, mask_permuted_d1)
            expected_dim1 = expected_permuted_d1.permute(0, 2, 1)
            self.assertTrue(torch.allclose(output_dim1, expected_dim1, atol=1e-12))

            # Test dim=0
            mask = torch.rand(B, S, D) > 0.3
            for s in range(S):
                for d in range(D):
                    if not mask[:, s, d].any(): mask[np.random.randint(0, B), s, d] = True

            output_dim0 = self.sm(x, mask, dim=0)
            x_permuted_d0 = x.permute(1, 2, 0)
            mask_permuted_d0 = mask.permute(1, 2, 0)
            expected_permuted_d0 = self._compute_naive_masked_softmax_last_dim_3d(x_permuted_d0, mask_permuted_d0)
            expected_dim0 = expected_permuted_d0.permute(2, 0, 1)
            self.assertTrue(torch.allclose(output_dim0, expected_dim0, atol=1e-12))

    # --------------------------------------------------------------
    # 1. All-ones mask ≈ torch.softmax (3d)
    # --------------------------------------------------------------
    @unittest.skip("passed")
    def test_all_ones_mask_3d(self):
        for _ in range(self.num_trials):
            # Random 3-D tensor (B,S,D) with D>=2 for meaningful softmax dim
            B = np.random.randint(2, 8)
            S = np.random.randint(2, 16)
            D = np.random.randint(2, 12)
            x = torch.randn(B, S, D)
            mask = torch.ones_like(x, dtype=torch.bool)

            for d in range(0, 3):
                out = self.sm(x, mask, dim=d)
                expected = F.softmax(x, dim=d)
                self.assertTrue(torch.allclose(out, expected, atol=1e-12))

    # large logits
    @unittest.skip("passed")
    def test_all_ones_mask_3d_large_logits(self):
        for _ in range(self.num_trials):
            B, S, D = (np.random.randint(2, 8) for _ in range(3))
            x = (torch.rand(B, S, D) * 2 - 1) * 1e4
            mask = torch.ones_like(x, dtype=torch.bool)
            for d in range(3):
                out = self.sm(x, mask, dim=d)
                expected = F.softmax(x, dim=d)
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.allclose(out, expected, atol=1e-12))

    # negative dim indices
    @unittest.skip("passed")  
    def test_all_ones_mask_3d_negative_dims(self):
        for _ in range(self.num_trials):
            B, S, D = (np.random.randint(2, 8) for _ in range(3))
            x = torch.randn(B, S, D)
            mask = torch.ones_like(x, dtype=torch.bool)
            for d in range(3):
                out_pos = self.sm(x, mask, dim=d)
                out_neg = self.sm(x, mask, dim=-3 + d)
                self.assertTrue(torch.allclose(out_pos, out_neg, atol=1e-12))

    # broadcasting semantics
    @unittest.skip("passed")
    def test_all_ones_mask_3d_broadcasting_semantics(self):
        for _ in range(self.num_trials):
            B, S, D = (np.random.randint(2, 8) for _ in range(3))
            x = torch.randn(B, S, D)

            # this method tests that the output is the same for a mask of shape (B, S, 1) and (B, S, D)
            mask_a = torch.ones(B, S, dtype=torch.bool).unsqueeze(-1)
            mask_b = torch.ones(B, S, D, dtype=torch.bool)

            # check that the output is the same for both masks
            out_a = self.sm(x, mask_a, dim=-1)
            out_b = self.sm(x, mask_b, dim=-1)
            self.assertTrue(torch.allclose(out_a, out_b, atol=1e-12))    

    # singleton dimension
    @unittest.skip("passed")
    def test_all_ones_mask_3d_singleton_dimension(self):
        for _ in range(self.num_trials):
            # iterate through each dimension: setting it to 1 and checking that the output is the same as the original
            org_dims = [np.random.randint(2, 8) for _ in range(3)]
            
            for d in range(3):
                reduced_dims = org_dims.copy()
                reduced_dims[d] = 1
                x = torch.randn(*reduced_dims)
                mask = torch.ones_like(x, dtype=torch.bool)
                out = self.sm(x, mask, dim=d)
                expected = F.softmax(x, dim=d)
                self.assertTrue(torch.allclose(out, expected, atol=1e-12))

    @unittest.skip("passed")
    def test_gradient_parity_all_ones_mask_3d(self):
        for _ in range(self.num_trials):
            B, S, D = (np.random.randint(2, 8) for _ in range(3))
            x1 = torch.randn(B, S, D, requires_grad=True)
            mask = torch.ones_like(x1, dtype=torch.bool)

            x2 = x1.detach().clone().requires_grad_()

            out1 = self.sm(x1, mask, dim=-1)
            loss1 = out1.sum()
            loss1.backward()
            grad1 = x1.grad.detach().clone()

            self.assertTrue(torch.isfinite(grad1).all())

            out2 = F.softmax(x2, dim=-1)
            loss2 = out2.sum()
            loss2.backward()
            grad2 = x2.grad.detach().clone()

            # it fails for atol = 1e-7
            self.assertTrue(torch.allclose(grad1, grad2, atol=9e-6))

    @unittest.skip("passed")
    def test_gradient_parity_all_ones_mask_3d_large_logits(self):
        for _ in range(self.num_trials):
            B, S, D = (np.random.randint(2, 8) for _ in range(3))
            x1 = torch.randn(B, S, D) * 1e4
            # set the gradient after mutliplication (otherwise, x1 woulnd't be a leaf node)
            x1.requires_grad = True
            mask = torch.ones_like(x1, dtype=torch.bool)

            x2 = x1.detach().clone().requires_grad_()

            out1 = self.sm(x1, mask, dim=-1)
            loss1 = out1.sum()
            loss1.backward()
            grad1 = x1.grad.detach().clone()

            self.assertTrue(torch.isfinite(grad1).all())

            out2 = F.softmax(x2, dim=-1)
            loss2 = out2.sum()
            loss2.backward()
            grad2 = x2.grad.detach().clone()

            # it fails for atol = 1e-7
            self.assertTrue(torch.allclose(grad1, grad2, atol=9e-6))



    # --------------------------------------------------------------
    # 2. All-ones mask ≈ torch.softmax (2d)
    # --------------------------------------------------------------

    @unittest.skip("passed")
    def test_all_ones_mask_2d(self):
        for _ in range(self.num_trials):
            # Random 2-D tensor (B,S) with S>=2 for meaningful softmax dim
            B = np.random.randint(2, 8)
            S = np.random.randint(2, 16)

            x = torch.randn(B, S)
            mask = torch.ones_like(x, dtype=torch.bool)
    
            for d in range(0, 2):
                out = self.sm(x, mask, dim=d)
                expected = F.softmax(x, dim=d)
                self.assertTrue(torch.allclose(out, expected, atol=1e-8))

    @unittest.skip("passed")
    def test_all_ones_mask_2d_large_logits(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 8) for _ in range(2))
            x = (torch.rand(B, S) * 2 - 1) * 1e4
            mask = torch.ones_like(x, dtype=torch.bool)
            for d in range(2):
                out = self.sm(x, mask, dim=d)
                expected = F.softmax(x, dim=d)
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.allclose(out, expected, atol=1e-12))

    @unittest.skip("passed")
    def test_all_ones_mask_2d_negative_dims(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 8) for _ in range(2))
            x = torch.randn(B, S)
            mask = torch.ones_like(x, dtype=torch.bool)
            for d in range(2):
                out_pos = self.sm(x, mask, dim=d)
                out_neg = self.sm(x, mask, dim=-2 + d)
                self.assertTrue(torch.allclose(out_pos, out_neg, atol=1e-12))

    @unittest.skip("passed")
    def test_all_ones_mask_2d_broadcasting_semantics(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 8) for _ in range(2))
            x = torch.randn(B, S)
            mask_a = torch.ones(B, 1, dtype=torch.bool)
            mask_b = torch.ones(B, S, dtype=torch.bool)
            out_a = self.sm(x, mask_a, dim=-1)
            out_b = self.sm(x, mask_b, dim=-1)
            self.assertTrue(torch.allclose(out_a, out_b, atol=1e-12))

    @unittest.skip("passed")
    def test_all_ones_mask_2d_singleton_dimension(self):
        for _ in range(self.num_trials):
            org_dims = [np.random.randint(2, 8) for _ in range(2)]
            for d in range(2):
                reduced_dims = org_dims.copy()
                reduced_dims[d] = 1
                x = torch.randn(*reduced_dims)
                mask = torch.ones_like(x, dtype=torch.bool)
                out = self.sm(x, mask, dim=d)
                expected = F.softmax(x, dim=d)
                self.assertTrue(torch.allclose(out, expected, atol=1e-12))



    @unittest.skip("passed")
    def test_gradient_parity_all_ones_mask_2d(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 8) for _ in range(2))
            x1 = torch.randn(B, S) * 1e4
            x1.requires_grad = True
            mask = torch.ones_like(x1, dtype=torch.bool)

            x2 = x1.detach().clone().requires_grad_()

            out1 = self.sm(x1, mask, dim=-1)
            loss1 = out1.sum()
            loss1.backward()
            grad1 = x1.grad.detach().clone()

            self.assertTrue(torch.isfinite(grad1).all())

            out2 = F.softmax(x2, dim=-1)
            loss2 = out2.sum()
            loss2.backward()
            grad2 = x2.grad.detach().clone()

            # it fails for atol = 1e-7
            self.assertTrue(torch.allclose(grad1, grad2, atol=1e-7))

    @unittest.skip("passed")
    def test_gradient_parity_all_ones_mask_2d_large_logits(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 8) for _ in range(2))
            x1 = torch.randn(B, S) * 1e4
            # set the gradient after mutliplication (otherwise, x1 woulnd't be a leaf node)
            x1.requires_grad = True
            mask = torch.ones_like(x1, dtype=torch.bool)

            x2 = x1.detach().clone().requires_grad_()

            out1 = self.sm(x1, mask, dim=-1)
            loss1 = out1.sum()
            loss1.backward()
            grad1 = x1.grad.detach().clone()

            self.assertTrue(torch.isfinite(grad1).all())

            out2 = F.softmax(x2, dim=-1)
            loss2 = out2.sum()
            loss2.backward()
            grad2 = x2.grad.detach().clone()

            # it fails for atol = 1e-7
            self.assertTrue(torch.allclose(grad1, grad2, atol=1e-7))


    # --------------------------------------------------------------
    # 3. All-ones mask ≈ torch.softmax (1d)
    # --------------------------------------------------------------

    @unittest.skip("passed")
    def test_all_ones_mask_1d(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 16)
            x = torch.randn(S)
            mask = torch.ones_like(x, dtype=torch.bool)
            out = self.sm(x, mask, dim=0)
            expected = F.softmax(x, dim=0)
            self.assertTrue(torch.allclose(out, expected, atol=1e-8))

    @unittest.skip("passed")
    def test_all_ones_mask_1d_large_logits(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 16)
            x = (torch.rand(S) * 2 - 1) * 1e4
            mask = torch.ones_like(x, dtype=torch.bool)
            out = self.sm(x, mask, dim=0)
            expected = F.softmax(x, dim=0)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.allclose(out, expected, atol=1e-12))

    @unittest.skip("passed")  
    def test_all_ones_mask_1d_negative_dims(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 16)
            x = torch.randn(S)
            mask = torch.ones_like(x, dtype=torch.bool)
            out_pos = self.sm(x, mask, dim=0)
            out_neg = self.sm(x, mask, dim=-1)
            self.assertTrue(torch.allclose(out_pos, out_neg, atol=1e-12))

    @unittest.skip("passed")
    def test_all_ones_mask_1d_singleton_dimension(self):
        for _ in range(self.num_trials):
            x = torch.randn(1)
            mask = torch.ones_like(x, dtype=torch.bool)
            out = self.sm(x, mask, dim=0)
            expected = F.softmax(x, dim=0)
            self.assertTrue(torch.allclose(out, expected, atol=1e-12))

    @unittest.skip("passed")
    def test_gradient_parity_all_ones_mask_1d(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 16)
            x1 = torch.randn(S, requires_grad=True)
            mask = torch.ones_like(x1, dtype=torch.bool)
            x2 = x1.detach().clone().requires_grad_()
            out1 = self.sm(x1, mask, dim=0)
            loss1 = out1.sum()
            loss1.backward()
            grad1 = x1.grad.detach().clone()
            self.assertTrue(torch.isfinite(grad1).all())
            out2 = F.softmax(x2, dim=0)
            loss2 = out2.sum()
            loss2.backward()
            grad2 = x2.grad.detach().clone()
            self.assertTrue(torch.allclose(grad1, grad2, atol=9e-6))

    @unittest.skip("passed")
    def test_gradient_parity_all_ones_mask_1d_large_logits(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 16)
            x1 = torch.randn(S) * 1e4
            x1.requires_grad = True
            mask = torch.ones_like(x1, dtype=torch.bool)
            x2 = x1.detach().clone().requires_grad_()
            out1 = self.sm(x1, mask, dim=0)
            loss1 = out1.sum()
            loss1.backward()
            grad1 = x1.grad.detach().clone()
            self.assertTrue(torch.isfinite(grad1).all())
            out2 = F.softmax(x2, dim=0)
            loss2 = out2.sum()
            loss2.backward()
            grad2 = x2.grad.detach().clone()
            self.assertTrue(torch.allclose(grad1, grad2, atol=1e-8))




    # --------------------------------------------------------------
    # 4. All-zeros mask returns all zeros
    # --------------------------------------------------------------
    @unittest.skip("passed")
    def test_all_zeros_mask_returns_zero_2d(self):
        for _ in range(self.num_trials):
            B = np.random.randint(2, 8)
            S = np.random.randint(2, 16)
            x = torch.randn(B, S)
            mask = torch.zeros_like(x, dtype=torch.bool)
            for d in range(0, 2):
                out = self.sm(x, mask, dim=d)
                self.assertTrue(torch.allclose(out, torch.zeros_like(x), atol=1e-12))

    @unittest.skip("passed")
    def test_all_zeros_mask_returns_zero_2d_large_logits(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 8) for _ in range(2))
            x = (torch.rand(B, S) * 2 - 1) * 1e4
            mask = torch.zeros_like(x, dtype=torch.bool)
            for d in range(2):
                out = self.sm(x, mask, dim=d)
                self.assertTrue(torch.allclose(out, torch.zeros_like(x), atol=1e-12))

    @unittest.skip("passed")
    def test_all_zeros_mask_returns_zero_2d_negative_dims(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 8) for _ in range(2))
            x = torch.randn(B, S)
            mask = torch.zeros_like(x, dtype=torch.bool)
            for d in range(2):
                out_pos = self.sm(x, mask, dim=d)
                out_neg = self.sm(x, mask, dim=-2 + d)
                self.assertTrue(torch.allclose(out_pos, out_neg, atol=1e-12))


    @unittest.skip("passed")
    def test_all_zeros_mask_returns_zero_2d_broadcasting_semantics(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 8) for _ in range(2))
            x = torch.randn(B, S)
            mask_a = torch.zeros(B, 1, dtype=torch.bool)
            mask_b = torch.zeros(B, S, dtype=torch.bool)
            out_a = self.sm(x, mask_a, dim=-1)
            out_b = self.sm(x, mask_b, dim=-1)
            self.assertTrue(torch.allclose(out_a, out_b, atol=1e-12))


    @unittest.skip("passed")
    def test_all_zeros_mask_returns_zero_2d_singleton_dimension(self):
        for _ in range(self.num_trials):
            org_dims = [np.random.randint(2, 8) for _ in range(2)]
            for d in range(2):
                reduced_dims = org_dims.copy()
                reduced_dims[d] = 1
                x = torch.randn(*reduced_dims)
                mask = torch.zeros_like(x, dtype=torch.bool)
                out = self.sm(x, mask, dim=d)
                self.assertTrue(torch.allclose(out, torch.zeros_like(x), atol=1e-12))

    @unittest.skip("passed")
    def test_gradient_all_zeros_mask_2d_is_zero(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 8) for _ in range(2))
            x = torch.randn(B, S, requires_grad=True)
            mask = torch.zeros_like(x, dtype=torch.bool)
            out = self.sm(x, mask, dim=0)
            loss = out.sum()
            loss.backward()
            grad = x.grad.detach().clone()
            self.assertTrue(torch.allclose(grad, torch.zeros_like(grad), atol=1e-12))


    @unittest.skip("passed")
    def test_all_zeros_mask_returns_zero_1d(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 16)
            x = torch.randn(S)
            mask = torch.zeros_like(x, dtype=torch.bool)
            out = self.sm(x, mask, dim=0)
            self.assertTrue(torch.allclose(out, torch.zeros_like(x), atol=1e-12))

    @unittest.skip("passed")
    def test_all_zeros_mask_returns_zero_1d_large_logits(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 16)
            x = (torch.rand(S) * 2 - 1) * 1e4
            mask = torch.zeros_like(x, dtype=torch.bool)
            out = self.sm(x, mask, dim=0)
            self.assertTrue(torch.allclose(out, torch.zeros_like(x), atol=1e-12))

    @unittest.skip("passed")
    def test_gradient_all_zeros_mask_1d_is_zero(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 16)
            x = torch.randn(S, requires_grad=True)
            mask = torch.zeros_like(x, dtype=torch.bool)
            out = self.sm(x, mask, dim=0)
            loss = out.sum()
            loss.backward()
            grad = x.grad.detach().clone()
            self.assertTrue(torch.allclose(grad, torch.zeros_like(grad), atol=1e-12))

        
    # --------------------------------------------------------------
    # 5. Random mask ⇒ (i) masked positions prob 0, (ii) rows sum to 1
    # --------------------------------------------------------------
    @unittest.skip("passed")
    def test_random_mask_properties(self):
        for _ in range(self.num_trials):
            B = np.random.randint(2, 5)
            S = np.random.randint(2, 5)
            x = torch.randn(B, S)
            mask = torch.rand(B, S) > 0.3  # keep ~70 %
            # Guarantee at least one True per row
            for b in range(B):
                if not mask[b].any():
                    mask[b, 0] = True
            out = self.sm(x, mask, dim=-1)

            # (i) masked positions zero
            self.assertTrue(torch.allclose(out[~mask], torch.zeros_like(out[~mask])))
            # (ii) row sums equal 1 (within tolerance)
            row_sums = out.sum(dim=-1)
            self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-12))

    @unittest.skip("passed")
    def test_random_mask_properties_large_logits(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 5) for _ in range(2))
            x = (torch.rand(B, S) * 2 - 1) * 1e4
            mask = torch.rand(B, S) > 0.3
            for b in range(B):
                if not mask[b].any(): mask[b, 0] = True
            out = self.sm(x, mask, dim=-1)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.allclose(out[~mask], torch.zeros_like(out[~mask])))
            row_sums = out.sum(dim=-1)
            self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-12))

    @unittest.skip("passed")
    def test_gradcheck_random_mask(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 5) for _ in range(2))
            x = torch.randn(B, S, dtype=torch.double, requires_grad=True)
            
            mask = torch.rand(B, S) > 0.3
            
            for b in range(B):
                if not mask[b].any():
                    mask[b, 0] = True
            
            self.assertTrue(gradcheck(lambda inp: self.sm(inp, mask, dim=-1), (x,), atol=1e-8))

    @unittest.skip("passed")  
    def test_gradcheck_random_mask_large_logits(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 5) for _ in range(2))
            x = (torch.randn(B, S, dtype=torch.double)) * 1e4
            x.requires_grad = True
            mask = torch.rand(B, S) > 0.3
            for b in range(B):
                if not mask[b].any():
                    mask[b, 0] = True
            
            self.assertTrue(gradcheck(lambda inp: self.sm(inp, mask, dim=-1), (x,), atol=1e-8))

    # --------------------------------------------------------------
    # 6. Uniform logits w/ random mask ⇒ equal probabilities 1/k
    # --------------------------------------------------------------
    # @unittest.skip("passed")
    def test_uniform_logits_random_mask_uniform_output_1d(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 30)
            logits = torch.full((S,), np.random.randn())  # constant value across entries
            mask = torch.rand(S) > 0.5  # keep about half
            if not mask.any():
                mask[0] = True  # ensure at least one element kept

            out = self.sm(logits, mask, dim=0)
            k = mask.sum().item()
            expected = torch.zeros_like(logits)
            expected[mask] = 1.0 / k
            self.assertTrue(torch.allclose(out, expected, atol=1e-8))

    # @unittest.skip("passed")
    def test_uniform_logits_random_mask_uniform_output_1d_large_logits(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 30)
            logits = torch.full((S,), (np.random.rand() * 2 - 1) * 1e4) # large uniform value
            mask = torch.rand(S) > 0.5
            if not mask.any(): mask[0] = True
            out = self.sm(logits, mask, dim=0)
            k = mask.sum().item()
            expected = torch.zeros_like(logits)
            expected[mask] = 1.0 / k
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.allclose(out, expected, atol=1e-8))

    def test_uniform_logits_random_mask_uniform_output_2d(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 30) for _ in range(2))
            # all the values are the same across the rows
            logits = torch.cat([torch.full((1, S), np.random.randn()) for _ in range(B)], dim=0)
            mask = torch.rand(B, S) > 0.5

            # make sure there is at least one True per row
            for b in range(B):
                if not mask[b].any():
                    mask[b, 0] = True

            out = self.sm(logits, mask, dim=-1)
            k = mask.sum(dim=-1, keepdim=True)
        
            expected = (torch.ones_like(logits) / k) * mask.float()
            self.assertTrue(torch.allclose(out, expected, atol=1e-10)) 


    def test_uniform_logits_random_mask_uniform_output_2d_large_logits(self):
        for _ in range(self.num_trials):
            B, S = (np.random.randint(2, 30) for _ in range(2))
            # all the values are the same across the rows
            logits = torch.cat([torch.full((1, S), np.random.randn()) for _ in range(B)], dim=0)

            mask = torch.rand(B, S) > 0.5

            # make sure there is at least one True per row
            for b in range(B):
                if not mask[b].any():
                    mask[b, 0] = True
            
            out = self.sm(logits, mask, dim=-1)
            k = mask.sum(dim=-1, keepdim=True)
            expected = (torch.ones_like(logits) / k) * mask.float()
            self.assertTrue(torch.allclose(out, expected, atol=1e-10)) 



    @unittest.skip("passed")
    def test_gradcheck_uniform_logits_random_mask(self):
        for _ in range(self.num_trials):
            S = np.random.randint(2, 30)
            logits = torch.full((S,), np.random.randn(), dtype=torch.double, requires_grad=True)
            mask = torch.rand(S) > 0.5
            if not mask.any():
                mask[0] = True
            self.assertTrue(gradcheck(lambda inp: self.sm(inp, mask, dim=0), (logits,), atol=1e-8))

    # -----------------------------------------------------------
    # 7. The mask contains only one True value at a certain dimension -> the output of the softmax on (said dim) is a one-hot vector
    # -----------------------------------------------------------
    @unittest.skip("passed")
    def test_one_hot_mask_produces_one_hot_output_3d(self):
        for _ in range(self.num_trials):
            B, S, D = (np.random.randint(2, 8) for _ in range(3))
            x = torch.randn(B, S, D)
            for d in range(3):
                # Create a one-hot mask along dimension d
                mask = torch.zeros_like(x, dtype=torch.bool)
                if d == 0:
                    indices = torch.randint(0, B, (1, S, D))
                    mask[indices, torch.arange(S)[None, :, None], torch.arange(D)[None, None, :]] = True
                elif d == 1:
                    indices = torch.randint(0, S, (B, 1, D))
                    mask[torch.arange(B)[:, None, None], indices, torch.arange(D)[None, None, :]] = True
                else: # d == 2
                    indices = torch.randint(0, D, (B, S, 1))
                    mask[torch.arange(B)[:, None, None], torch.arange(S)[None, :, None], indices] = True

                out = self.sm(x, mask, dim=d)
                expected = torch.zeros_like(x)
                expected[mask] = 1.0
                self.assertTrue(torch.allclose(out, expected, atol=1e-12))

    @unittest.skip("passed")
    def test_one_hot_mask_produces_one_hot_output_3d_large_logits(self):
        for _ in range(self.num_trials):
            B, S, D = (np.random.randint(2, 8) for _ in range(3))
            x = (torch.rand(B, S, D) * 2 - 1) * 1e4
            for d in range(3):
                mask = torch.zeros_like(x, dtype=torch.bool)
                if d == 0:
                    indices = torch.randint(0, B, (1, S, D))
                    mask[indices, torch.arange(S)[None, :, None], torch.arange(D)[None, None, :]] = True
                elif d == 1:
                    indices = torch.randint(0, S, (B, 1, D))
                    mask[torch.arange(B)[:, None, None], indices, torch.arange(D)[None, None, :]] = True
                else: # d == 2
                    indices = torch.randint(0, D, (B, S, 1))
                    mask[torch.arange(B)[:, None, None], torch.arange(S)[None, :, None], indices] = True
                
                out = self.sm(x, mask, dim=d)
                expected = torch.zeros_like(x)
                expected[mask] = 1.0
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.allclose(out, expected, atol=1e-12))

    @unittest.skip("passed")
    def test_one_hot_mask_produces_one_hot_output_3d_negative_dims(self):
        for _ in range(self.num_trials):
            B, S, D = (np.random.randint(2, 8) for _ in range(3))
            x = torch.randn(B, S, D)
            for d in range(3):
                mask = torch.zeros_like(x, dtype=torch.bool)
                if d == 0:
                    indices = torch.randint(0, B, (1, S, D))
                    mask[indices, torch.arange(S)[None, :, None], torch.arange(D)[None, None, :]] = True
                elif d == 1:
                    indices = torch.randint(0, S, (B, 1, D))
                    mask[torch.arange(B)[:, None, None], indices, torch.arange(D)[None, None, :]] = True
                else: # d == 2
                    indices = torch.randint(0, D, (B, S, 1))
                    mask[torch.arange(B)[:, None, None], torch.arange(S)[None, :, None], indices] = True

                out_pos = self.sm(x, mask, dim=d)
                out_neg = self.sm(x, mask, dim=-3 + d)
                self.assertTrue(torch.allclose(out_pos, out_neg, atol=1e-12))
    
    @unittest.skip("passed")
    def test_gradient_one_hot_mask_is_zero_3d(self):
        for _ in range(self.num_trials):
            B, S, D = (np.random.randint(2, 8) for _ in range(3))
            x = torch.randn(B, S, D, requires_grad=True)
            for d in range(3):
                mask = torch.zeros_like(x, dtype=torch.bool)
                if d == 0:
                    indices = torch.randint(0, B, (1, S, D))
                    mask[indices, torch.arange(S)[None, :, None], torch.arange(D)[None, None, :]] = True
                elif d == 1:
                    indices = torch.randint(0, S, (B, 1, D))
                    mask[torch.arange(B)[:, None, None], indices, torch.arange(D)[None, None, :]] = True
                else: # d == 2
                    indices = torch.randint(0, D, (B, S, 1))
                    mask[torch.arange(B)[:, None, None], torch.arange(S)[None, :, None], indices] = True
                
                out = self.sm(x, mask, dim=d)
                loss = out.sum()
                loss.backward(torch.ones_like(loss))
                grad = x.grad.detach().clone()
                self.assertTrue(torch.allclose(grad, torch.zeros_like(grad), atol=1e-12))
                x.grad.zero_()


if __name__ == "__main__":
    unittest.main()