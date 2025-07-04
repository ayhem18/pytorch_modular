import torch
import unittest
import numpy as np

from tqdm import tqdm
from typing import Tuple

from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.attention.attention_layer import SingleHeadAttentionLayer
from tests.building_blocks.attention.naive_implementations.naive_single_head_att import NaiveSHA

class TestSingleHeadAttentionLayer(CustomModuleBaseTest):
    def setUp(self):
        # Define common dimensions for testing
        self.input_dim = 64
        self.key_dim = 16 # a square 
        self.value_dim = 16 
        
        # Number of test iterations
        self.num_iterations = 1000
        
        # Create both implementations
        self.vectorized_att = SingleHeadAttentionLayer(
            input_dimension=self.input_dim,
            value_dim=self.value_dim,
            key_dim=self.key_dim
        )
        
        self.naive_att = NaiveSHA(
            input_dimension=self.input_dim,
            value_dim=self.value_dim,
            key_dim=self.key_dim
        )
        
        # Set both to the same random weights
        self._sync_weights()
        
    def _sync_weights(self):
        """Synchronize weights between vectorized and naive implementations"""
        # For W_q
        self.naive_att.W_q.weight.data = self.vectorized_att.W_q.weight.data.clone()
        self.naive_att.W_q.bias.data = self.vectorized_att.W_q.bias.data.clone()
        
        # For W_k
        self.naive_att.W_k.weight.data = self.vectorized_att.W_k.weight.data.clone()
        self.naive_att.W_k.bias.data = self.vectorized_att.W_k.bias.data.clone()
        
        # For W_v
        self.naive_att.W_v.weight.data = self.vectorized_att.W_v.weight.data.clone()
        self.naive_att.W_v.bias.data = self.vectorized_att.W_v.bias.data.clone()
        
        # For W_o
        self.naive_att.W_o.weight.data = self.vectorized_att.W_o.weight.data.clone()
        self.naive_att.W_o.bias.data = self.vectorized_att.W_o.bias.data.clone()
    
    def _generate_random_input(self, batch_size: int, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random input, query, key and value tensors"""
        torch_dtype = self.vectorized_att.W_q.weight.dtype
        x = torch.randn(batch_size, seq_length, self.input_dim).to(torch_dtype)
        

        # Apply linear transformations to get q, k, v
        with torch.no_grad():
            q = self.vectorized_att.W_q(x)
            k = self.vectorized_att.W_k(x)
            v = self.vectorized_att.W_v(x)
        
        assert q.shape == (batch_size, seq_length, self.key_dim)
        assert k.shape == (batch_size, seq_length, self.key_dim)
        assert v.shape == (batch_size, seq_length, self.value_dim)  

        return x, q, k, v

    @unittest.skip("passed")
    def test_causal_mask_creation(self):
        """Test that both implementations create the same attention mask"""
        for _ in tqdm(range(self.num_iterations), desc="Testing attention mask creation"):
            # Generate a random sequence length between 5 and 50
            seq_length = np.random.randint(5, 50)
            batch_size = np.random.randint(1, 10)
            
            # Generate masks from both implementations
            vec_mask = self.vectorized_att._causal_attention_mask(batch_size, seq_length)
            naive_mask = self.naive_att._causal_attention_mask(batch_size, seq_length)
        
            # Shape assertions
            self.assertEqual(vec_mask.shape, (batch_size, seq_length, seq_length))
            self.assertEqual(naive_mask.shape, (batch_size, seq_length, seq_length))

            # make sure the mask is of the same shape
            self.assertTrue(torch.all(vec_mask == naive_mask))
            # make sure all batch elements in the mask are the same
            for b in range(batch_size):
                self.assertTrue(torch.allclose(vec_mask[b], vec_mask[0]))


    @unittest.skip("passed")
    def test_query_key_product(self):
        """Test that both implementations compute the same query-key product"""
        for _ in tqdm(range(self.num_iterations), desc="Testing query-key product"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            
            # Generate random q and k tensors
            _, q, k, _ = self._generate_random_input(batch_size, seq_length)
            
            # Compute query-key product using both implementations
            vec_product = self.vectorized_att._key_query_product(q, k)
            naive_product = self.naive_att._key_query_product(q, k)
            
            # Check they have the same shape
            self.assertEqual(vec_product.shape, naive_product.shape)
            
            # Check that they are approximately equal
            self.assertTrue(torch.allclose(vec_product, naive_product, atol=2*1e-7))

    @unittest.skip("passed")
    def test_compute_weights(self):
        """Test that both implementations compute the same attention weights"""
        for _ in tqdm(range(self.num_iterations), desc="Testing compute weights"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)

            # Generate random q and k tensors
            _, q, k, _ = self._generate_random_input(batch_size, seq_length)

            # Generate query-key product and mask
            vec_product = self.vectorized_att._key_query_product(q, k)
            naive_product = self.naive_att._key_query_product(q, k)
            
            vec_mask = self.vectorized_att._causal_attention_mask(batch_size, seq_length)
            naive_mask = self.naive_att._causal_attention_mask(batch_size, seq_length)
            
            final_vec_mask = self.vectorized_att.create_final_mask(vec_mask, None)
            final_naive_mask = self.naive_att.create_final_mask(naive_mask, None)

            # inf_naive_mask = self.naive_att._process_final_mask(final_naive_mask)

            # Compute weights using both implementations
            vec_weights = self.vectorized_att._compute_weights(vec_product, final_vec_mask)
            naive_weights = self.naive_att._compute_weights(naive_product, final_naive_mask)
            
            # Check they have the same shape
            self.assertEqual(vec_weights.shape, naive_weights.shape)
            
            # Check that they are approximately equal
            self.assertTrue(torch.allclose(vec_weights, naive_weights, atol=1e-10))

            # Additional check: make sure masked positions (j > i) are ~0 after soft-max
            for b in range(batch_size):
                for i in range(seq_length):
                    num_zeros = (vec_weights[b, i, i+1:] < 1e-9).sum().item()
                    self.assertEqual(
                        num_zeros,
                        seq_length - (i + 1),
                        msg=f"Row {i} in batch {b} should have {seq_length - (i + 1)} zeros, found {num_zeros}"
                    )

    @unittest.skip("passed")
    def test_compute_weighted_values(self):
        """Test that both implementations compute the same weighted values"""
        for _ in tqdm(range(self.num_iterations), desc="Testing compute weighted values"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            
            # Generate random tensors
            _, q, k, v = self._generate_random_input(batch_size, seq_length)
            
            # Generate query-key product and mask
            vec_product = self.vectorized_att._key_query_product(q, k)
            naive_product = self.naive_att._key_query_product(q, k)
            
            vec_mask = self.vectorized_att._causal_attention_mask(batch_size, seq_length)
            naive_mask = self.naive_att._causal_attention_mask(batch_size, seq_length)
            
            # Compute weights
            vec_weights = self.vectorized_att._compute_weights(vec_product, vec_mask)

            # naive_mask = self.naive_att._process_final_mask(naive_mask)
            naive_weights = self.naive_att._compute_weights(naive_product, naive_mask)
            
            # Compute weighted values
            vec_output = self.vectorized_att._compute_new_v(vec_weights, v)
            naive_output = self.naive_att._compute_new_v(naive_weights, v)
            
            # Check they have the same shape
            self.assertEqual(vec_output.shape, naive_output.shape)
            
            # Check that they are approximately equal
            self.assertTrue(torch.allclose(vec_output, naive_output, atol=1e-10))

    @unittest.skip("passed")
    def test_full_forward_pass(self):
        """Test that both implementations produce the same output for the full forward pass"""
        for _ in tqdm(range(self.num_iterations), desc="Testing full forward pass"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            
            # Generate random input tensor
            x, _, _, _ = self._generate_random_input(batch_size, seq_length)
            
            # Compute full forward pass
            with torch.no_grad():
                vec_output = self.vectorized_att.forward(x)
                naive_output = self.naive_att.forward(x)
            
            # Check they have the same shape
            self.assertEqual(vec_output.shape, naive_output.shape)
            
            # Check that they are approximately equal
            self.assertTrue(torch.allclose(vec_output, naive_output, atol=2*1e-7))


    ####   Custom Module Base Test Methods   ####
    @unittest.skip("passed")
    def test_eval_mode(self) -> None:
        """Test that calling eval() sets training=False for all parameters and submodules"""
        super()._test_eval_mode(self.vectorized_att)

    @unittest.skip("passed")
    def test_train_mode(self) -> None:
        """Test that calling train() sets training=True for all parameters and submodules"""
        super()._test_train_mode(self.vectorized_att)
    
    @unittest.skip("passed")
    def test_consistent_output_without_dropout_bn(self) -> None:
        """
        Test that modules without dropout or batch normalization 
        produce consistent output for the same input
        """
        for _ in range(self.num_iterations):
            input_tensor, _, _, _ = self._generate_random_input(10, 10)
            super()._test_consistent_output_without_dropout_bn(self.vectorized_att, input_tensor)
    
    @unittest.skip("passed")
    def test_consistent_output_in_eval_mode(self) -> None:
        """Test that all modules in eval mode produce consistent output for the same input"""
        for _ in range(self.num_iterations):
            input_tensor, _, _, _ = self._generate_random_input(10, 10)
            super()._test_consistent_output_in_eval_mode(self.vectorized_att, input_tensor)
    
    @unittest.skip("passed")
    def test_batch_size_one_in_train_mode(self) -> None:
        """
        Test that modules with batch normalization layers might raise errors 
        with batch size 1 in train mode
        """
        for _ in range(self.num_iterations):
            input_tensor, _, _, _ = self._generate_random_input(1, 10)
            super()._test_batch_size_one_in_train_mode(self.vectorized_att, input_tensor)

    
    @unittest.skip("passed")
    def test_batch_size_one_in_eval_mode(self) -> None:
        """Test that modules in eval mode should not raise errors for batch size 1"""
        for _ in range(self.num_iterations):
            input_tensor, _, _, _ = self._generate_random_input(1, 10)
            super()._test_batch_size_one_in_eval_mode(self.vectorized_att, input_tensor)

    @unittest.skip("passed")
    def test_named_parameters_length(self) -> None:
        """Test that named_parameters() and parameters() have the same length"""
        super()._test_named_parameters_length(self.vectorized_att)
    
    @unittest.skip("passed")
    def test_to_device(self) -> None:
        """Test that module can move between devices properly"""
        for _ in range(self.num_iterations):
            input_tensor, _, _, _ = self._generate_random_input(10, 10)
            super()._test_to_device(self.vectorized_att, input_tensor)

    # ------------------------------------------------------------------
    # tests that rely on *random* attention masks
    # ------------------------------------------------------------------
    def _generate_random_pad_mask(self, batch_size: int, seq_length: int):
        """Return random binary padding mask (B,S) where 1 keeps, 0 masks."""
        pad_mask = (torch.rand(batch_size, seq_length) > 0.3).int()  # ~70% keep
        # Ensure at least one token is kept per sequence
        for b in range(batch_size):
            if pad_mask[b].sum() == 0:
                pad_mask[b, 0] = 1
        return pad_mask

    @unittest.skip("passed")
    def test_create_final_mask(self):
        """Verify that create_final_mask obeys causal + pad constraints."""
        for _ in tqdm(range(self.num_iterations), desc="Testing final mask creation"):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)

            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

            causal_vec = self.vectorized_att._causal_attention_mask(batch_size, seq_length)
            causal_naive = self.naive_att._causal_attention_mask(batch_size, seq_length)

            vec_final = self.vectorized_att.create_final_mask(causal_vec, pad_mask)
            naive_final = self.naive_att.create_final_mask(causal_naive, pad_mask)

            # 1. Implementations agree
            self.assertTrue(torch.equal(vec_final, naive_final), "different masks from the different implementations")

            # 2. Causal property & pad property hold
            for b in range(batch_size):
                for i in range(seq_length):
                    for j in range(seq_length):
                        allowed = vec_final[b, i, j].item()
                        if j > i or not(pad_mask[b, i]) or not(pad_mask[b, j]):
                            self.assertFalse(allowed)
                        else:
                            self.assertTrue(allowed)

    @unittest.skip("passed")
    def test_compute_weights_with_random_mask(self):
        """Verify compute_weights produces identical outputs given an arbitrary user-supplied mask."""
        for _ in tqdm(range(self.num_iterations), desc="Testing compute weights w/ random masks"):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)

            _, q, k, _ = self._generate_random_input(batch_size, seq_length)

            vec_product = self.vectorized_att._key_query_product(q, k)
            naive_product = self.naive_att._key_query_product(q, k)

            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

            causal_vec = self.vectorized_att._causal_attention_mask(batch_size, seq_length)
            causal_naive = self.naive_att._causal_attention_mask(batch_size, seq_length)

            vec_final = self.vectorized_att.create_final_mask(causal_vec, pad_mask)
            naive_final = self.naive_att.create_final_mask(causal_naive, pad_mask)

            # inf_vec_mask = self.vectorized_att._process_final_mask(vec_final)
            # inf_naive_mask = self.naive_att._process_final_mask(naive_final)

            self.assertTrue(torch.allclose(vec_final, naive_final), "different masks from the different implementations")

            vec_weights = self.vectorized_att._compute_weights(vec_product, vec_final)
            naive_weights = self.naive_att._compute_weights(naive_product, naive_final)

            self.assertEqual(vec_weights.shape, naive_weights.shape)

            passed = False
            for i in range(1, 6):
                try:
                    self.assertTrue(
                    torch.allclose(vec_weights, naive_weights, atol=i * 1e-10),
                    msg="Vectorised and naive weights diverge under a random mask."
                    )
                    passed = True
                    break
                except AssertionError:
                    pass
            
            self.assertTrue(passed, msg="Vectorised and naive weights diverge under a random mask.")

            # 2. Causal property & pad property hold
            for b in range(batch_size):
                for i in range(seq_length):
                    for j in range(seq_length):
                        weight = vec_weights[b, i, j].item()
                        if j > i or pad_mask[b, i] == 0 or pad_mask[b, j] == 0:
                            self.assertTrue(weight < 1e-9, msg="the weight should be 0 for masked positions")


        

    @unittest.skip("passed")
    def test_compute_weighted_values(self):
        """Test that both implementations compute the same weighted values"""
        for _ in tqdm(range(self.num_iterations), desc="Testing compute weighted values"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            
            # Generate random tensors
            _, q, k, v = self._generate_random_input(batch_size, seq_length)
            
            # Generate query-key product and mask
            vec_product = self.vectorized_att._key_query_product(q, k)
            naive_product = self.naive_att._key_query_product(q, k)


            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)    
            causal_vec = self.vectorized_att._causal_attention_mask(batch_size, seq_length)
            final_mask = self.vectorized_att.create_final_mask(causal_vec, pad_mask)

            # Compute weights
            # inf_vec_mask = self.vectorized_att._process_final_mask(final_mask)
            # inf_naive_mask = self.naive_att._process_final_mask(final_mask)

            vec_weights = self.vectorized_att._compute_weights(vec_product, final_mask)
            naive_weights = self.naive_att._compute_weights(naive_product, final_mask)
            
            # Compute weighted values
            vec_output = self.vectorized_att._compute_new_v(vec_weights, v)
            naive_output = self.naive_att._compute_new_v(naive_weights, v)
            
            # Check they have the same shape
            self.assertEqual(vec_output.shape, naive_output.shape)
            
            passed = False
            for i in range(1, 6):
                try:
                    self.assertTrue(
                    torch.allclose(vec_output, naive_output, atol=i * 1e-7),
                    msg="Vectorised and naive outputs diverge under a random mask."
                    )
                    passed = True
                    break
                except AssertionError:
                    pass
            
            self.assertTrue(passed, msg="Vectorised and naive weights diverge under a random mask.")




    @unittest.skip("passed")
    def test_full_forward_pass_with_random_mask(self):
        """Test that both implementations produce the same output for the full forward pass"""
        for _ in tqdm(range(self.num_iterations), desc="Testing full forward pass w/ random mask"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            
            # Generate random input tensor
            x, _, _, _ = self._generate_random_input(batch_size, seq_length)
            
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

            # Compute full forward pass
            with torch.no_grad():
                vec_output = self.vectorized_att.forward(x, pad_mask)
                naive_output = self.naive_att.forward(x, pad_mask)
            
            # Check they have the same shape
            self.assertEqual(vec_output.shape, naive_output.shape)
            
            passed = False
            for i in range(1, 6):
                try:
                    self.assertTrue(
                    torch.allclose(vec_output, naive_output, atol=i * 1e-7),
                    msg="Vectorised and naive outputs diverge under a random mask."
                    )
                    passed = True
                    break
                except AssertionError:
                    pass
            
            self.assertTrue(passed, msg="Vectorised and naive outputs diverge under a random mask.")

    # @unittest.skip("passed")
    def test_gradcheck(self) -> None:
        """Test gradient computation using torch.autograd.gradcheck."""
        # Use small dimensions for speed
        for _ in tqdm(range(100), desc="Testing gradcheck"):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor, _, _, _ = self._generate_random_input(batch_size, seq_length)
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)
            
            # Test with mask
            super()._test_gradcheck(self.vectorized_att, input_tensor, pad_mask)
            # Test without mask
            super()._test_gradcheck(self.vectorized_att, input_tensor)

    # @unittest.skip("passed")
    def test_gradcheck_large_values(self) -> None:
        """Test gradient computation with large input values."""
        # Use small dimensions for speed
        for _ in tqdm(range(100), desc="Testing gradcheck with large values"):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor, _, _, _ = self._generate_random_input(batch_size, seq_length)
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)
        
            # Test with mask
            super()._test_gradcheck_large_values(self.vectorized_att, input_tensor, pad_mask)
            # Test without mask
            super()._test_gradcheck_large_values(self.vectorized_att, input_tensor)

    # @unittest.skip("passed")
    def test_grad_against_nan(self) -> None:
        """Test that the gradient is not nan"""
        for _ in range(self.num_iterations):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor, _, _, _ = self._generate_random_input(batch_size, seq_length)
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)
            super()._test_grad_against_nan(self.vectorized_att, input_tensor, pad_mask)
            super()._test_grad_against_nan(self.vectorized_att, input_tensor)


    # @unittest.skip("passed")
    def test_grad_against_nan_large_values(self) -> None:
        """Test that the gradient is not nan with large input values"""
        for _ in range(self.num_iterations):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor, _, _, _ = self._generate_random_input(batch_size, seq_length)
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)
            super()._test_grad_against_nan_large_values(self.vectorized_att, input_tensor, pad_mask)
            super()._test_grad_against_nan_large_values(self.vectorized_att, input_tensor)


if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(0)
    unittest.main()
