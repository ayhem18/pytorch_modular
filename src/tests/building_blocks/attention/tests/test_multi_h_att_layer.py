import abc
import torch
import unittest
import numpy as np

from tqdm import tqdm

from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.attention.multi_head_att import AbstractMHAttentionLayer, CausalMultiHeadAttentionLayer, BidirectionalMultiHeadAttentionLayer
from tests.building_blocks.attention.naive_implementations.naive_multi_head_att import CausalNaiveMHA, BidirectionalNaiveMHA, AbstractNaiveMHA



class TestMultiHeadAttentionLayer(CustomModuleBaseTest, abc.ABC):

    @abc.abstractmethod
    def setUp(self):    
        self.vectorized_att: AbstractMHAttentionLayer = None
        self.naive_att: AbstractNaiveMHA = None
        
        pass

    def _sync_weights(self):
        """Synchronize weights between vectorized and naive implementations"""
        self.naive_att.sync_weights_from_vectorized(self.vectorized_att)
    
    def _generate_random_input(self, batch_size: int, seq_length: int) -> torch.Tensor:
        """Generate random input tensor"""
        return torch.randn(batch_size, seq_length, self.d_model)
    
    # ------------------------------------------------------------------
    # Helper for random padding masks
    # ------------------------------------------------------------------

    def _generate_random_pad_mask(self, batch_size: int, seq_length: int):
        """Return binary mask (B,S) where 1 keeps token, 0 masks."""
        pad_mask = (torch.rand(batch_size, seq_length) > 0.3).int()
        # ensure at least one token kept per sequence
        for b in range(batch_size):
            if pad_mask[b].sum() == 0:
                pad_mask[b, 0] = 1
        return pad_mask
    
    def _test_default_mask_creation(self):
        """Both implementations create identical causal boolean masks."""
        for _ in tqdm(range(self.num_iterations), desc="Testing causal mask creation"):
            seq_length = np.random.randint(5, 20)
            batch_size = np.random.randint(1, 5)

            vec_mask = self.vectorized_att._default_mask(batch_size, seq_length)
            naive_mask = self.naive_att._default_mask(batch_size, seq_length)           

            self.assertTrue(torch.equal(vec_mask, naive_mask))

            for b in range(batch_size):
                self.assertTrue(torch.equal(vec_mask[b], vec_mask[0]))


    def _test_query_key_product(self):
        """Test that both implementations compute the same query-key product"""
        for _ in tqdm(range(self.num_iterations), desc="Testing query-key product"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)
            
            # Generate random input
            x = self._generate_random_input(batch_size, seq_length)
            
            # Process through the first part of both implementations to get q and k
            with torch.no_grad():
                # For vectorized implementation
                q_vec = self.vectorized_att.W_q.forward(x)
                k_vec = self.vectorized_att.W_k.forward(x)
                q_vec = q_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
                k_vec = k_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
                
                # For naive implementation, we need to process each head separately
                q_naive_list = [None for _ in range(self.num_heads)]
                k_naive_list = [None for _ in range(self.num_heads)]
                for h in range(self.num_heads):
                    q_naive_list[h] = self.naive_att.W_q_list[h].forward(x)
                    k_naive_list[h] = self.naive_att.W_k_list[h].forward(x)
            
            # Test query-key product for the vectorized implementation
            vec_product = self.vectorized_att._key_query_product(q_vec, k_vec)
            
            # Test for each head in the naive implementation
            for h in range(self.num_heads):
                naive_product = self.naive_att._key_query_product(q_naive_list[h], k_naive_list[h])
                
                # Check shape match for this head
                self.assertEqual(vec_product[:, h].shape, naive_product.shape)
                
                # Check values are approximately equal
                validated = False
                for tol_coeff in range(1, 6):
                    try:
                        self.assertTrue(torch.allclose(vec_product[:, h], naive_product, atol=tol_coeff*1e-7))
                        validated = True
                        break
                    except:
                        continue

                self.assertTrue(validated, msg="Vectorised and naive query-key products diverge")


    # def _test_compute_weights(self):
    #     """Test that both implementations compute the same attention weights"""
    #     for _ in tqdm(range(self.num_iterations), desc="Testing compute weights"):
    #         # Generate random batch size and sequence length
    #         batch_size = np.random.randint(1, 5)
    #         seq_length = np.random.randint(5, 15)
            
    #         # Generate random input
    #         x = self._generate_random_input(batch_size, seq_length)
            
    #         # Process through both implementations to get query-key products
    #         with torch.no_grad():
    #             # For vectorized implementation
    #             q_vec = self.vectorized_att.W_q(x)
    #             k_vec = self.vectorized_att.W_k(x)
    #             q_vec = q_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    #             k_vec = k_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    #             vec_product = self.vectorized_att._key_query_product(q_vec, k_vec)
                
    #             # Create mask
    #             vec_mask = self.vectorized_att._default_mask(batch_size, seq_length)
    #             vec_mask = self.vectorized_att.create_final_mask(vec_mask, None)
    #             # vec_mask = self.vectorized_att._process_final_mask(vec_mask)
                
    #             # For naive implementation, process each head separately
    #             q_naive_list = [None for _ in range(self.num_heads)]
    #             k_naive_list = [None for _ in range(self.num_heads)]
    #             for h in range(self.num_heads):
    #                 q_naive_list[h] = self.naive_att.W_q_list[h].forward(x)
    #                 k_naive_list[h] = self.naive_att.W_k_list[h].forward(x)
                
    #             naive_mask = self.naive_att._default_mask(batch_size, seq_length)
    #             naive_mask = self.naive_att.create_final_mask(naive_mask, None)
    #             # naive_mask = self.naive_att._process_final_mask (naive_mask)

    #         self.assertTrue(torch.equal(vec_mask, naive_mask))

    #         # Compute weights using vectorized implementation
    #         vec_weights = self.vectorized_att._compute_weights(vec_product, vec_mask)
            
    #         # Test for each head in the naive implementation
    #         for h in range(self.num_heads):
    #             naive_product = self.naive_att._key_query_product(q_naive_list[h], k_naive_list[h])
    #             naive_weights = self.naive_att._compute_weights(naive_product, naive_mask[:, h])
                
    #             # Check shape match for this head       
    #             self.assertEqual(vec_weights[:, h].shape, naive_weights.shape)

    #             # Check values are approximately equal
    #             validated = False
    #             for tol_coeff in range(1, 6):
    #                 try:
    #                     self.assertTrue(torch.allclose(vec_weights[:, h], naive_weights, atol=tol_coeff*1e-7)) # can tolerate up to 5e-7
    #                     validated = True
    #                     break
    #                 except:
    #                     continue

    #             self.assertTrue(validated, msg="Vectorised and naive weights diverge")

    #         # at this point we know both implementations produce the same weights
    #         # so we can test the weights for the masked positions
    #         for b in range(batch_size):
    #             for i in range(seq_length):
    #                 for j in range(seq_length):
    #                     if j > i:
    #                         self.assertTrue(torch.all(vec_weights[b, :, i, j] < 1e-9), msg="the weight should be 0 for masked positions")


    def _test_compute_weighted_values(self):
        """Test that both implementations compute the same weighted values"""
        for _ in tqdm(range(self.num_iterations), desc="Testing compute weighted values"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)
            
            # Generate random input
            x = self._generate_random_input(batch_size, seq_length)
            
            # Process through both implementations
            with torch.no_grad():
                # For vectorized implementation
                q_vec = self.vectorized_att.W_q(x)
                k_vec = self.vectorized_att.W_k(x)
                v_vec = self.vectorized_att.W_v(x)
                
                q_vec = q_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
                k_vec = k_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
                v_vec = v_vec.view(batch_size, seq_length, self.num_heads, self.value_dim).permute(0, 2, 1, 3)
                
                vec_product = self.vectorized_att._key_query_product(q_vec, k_vec)
                
                vec_mask = self.vectorized_att._default_mask(batch_size, seq_length)
                vec_mask = self.vectorized_att.create_final_mask(vec_mask, None)
                
                vec_weights = self.vectorized_att._compute_weights(vec_product, vec_mask)
                
                # For naive implementation, process each head separately
                q_naive_list = [None for _ in range(self.num_heads)]
                k_naive_list = [None for _ in range(self.num_heads)]
                v_naive_list = [None for _ in range(self.num_heads)]

                for h in range(self.num_heads):
                    q_naive_list[h] = self.naive_att.W_q_list[h].forward(x)
                    k_naive_list[h] = self.naive_att.W_k_list[h].forward(x)
                    v_naive_list[h] = self.naive_att.W_v_list[h].forward(x)
                
                naive_mask = self.naive_att._default_mask(batch_size, seq_length)
                naive_mask = self.naive_att.create_final_mask(naive_mask, None)
                # naive_mask = self.naive_att._process_final_mask(naive_mask)
            
            self.assertTrue(torch.equal(vec_mask, naive_mask))

            # Compute weighted values using vectorized implementation
            vec_output = self.vectorized_att._compute_new_v(vec_weights, v_vec)
            
            # Test for each head in the naive implementation
            for h in range(self.num_heads):
                naive_product = self.naive_att._key_query_product(q_naive_list[h], k_naive_list[h])
                naive_weights = self.naive_att._compute_weights(naive_product, naive_mask[:, h])
                naive_output = self.naive_att._compute_new_v(naive_weights, v_naive_list[h])
                
                # Check shape match for this head
                self.assertEqual(vec_output[:, h].shape, naive_output.shape)
                
                # Check values are approximately equal
                validated = False
                for tol_coeff in range(1, 6):
                    try:
                        self.assertTrue(torch.allclose(vec_output[:, h], naive_output, atol=tol_coeff*1e-7))
                        validated = True
                        break
                    except:
                        continue

                self.assertTrue(validated, msg="Vectorised and naive outputs diverge")


    def _test_full_forward_pass(self):
        """Test that both implementations produce the same output for the full forward pass"""
        for _ in tqdm(range(self.num_iterations), desc="Testing full forward pass"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)
            
            # Generate random input tensor
            x = self._generate_random_input(batch_size, seq_length)
            
            # Compute full forward pass
            with torch.no_grad():
                vec_output = self.vectorized_att(x)
                naive_output = self.naive_att(x)
            
            # Check they have the same shape
            self.assertEqual(vec_output.shape, naive_output.shape)
            
            # Check that they are approximately equal
            validated = False
            for tol_coeff in range(1, 10):
                try:
                    self.assertTrue(torch.allclose(vec_output, naive_output, atol=tol_coeff*1e-7)) # can tolerate up to 9*1e-7
                    validated = True
                    break
                except:
                    continue

            self.assertTrue(validated, msg="Vectorised and naive outputs diverge")


    # ------------------------------------------------------------------
    # Tests with RANDOM padding masks
    # ------------------------------------------------------------------
    # def _test_create_final_mask_with_pad(self):
    #     """final_mask from both implementations agrees and respects causal+pad."""
    #     for _ in tqdm(range(self.num_iterations), desc="Testing final mask creation w/ pad"):
    #         batch_size = np.random.randint(1, 5)
    #         seq_length = np.random.randint(5, 15)

    #         pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

    #         causal_vec = self.vectorized_att._default_mask(batch_size, seq_length)
    #         causal_naive = self.naive_att._default_mask(batch_size, seq_length)

    #         vec_final = self.vectorized_att.create_final_mask(causal_vec, pad_mask)
    #         naive_final = self.naive_att.create_final_mask(causal_naive, pad_mask)

    #         self.assertTrue(torch.equal(vec_final, naive_final))

    #         # verify causal+pad rules
    #         for b in range(batch_size):
    #             for i in range(seq_length):
    #                 for j in range(seq_length):
    #                     if j > i or not pad_mask[b, i] or not pad_mask[b, j]:
    #                         self.assertFalse(torch.any(vec_final[b, :, i, j]))
    #                     else:
    #                         self.assertTrue(torch.all(vec_final[b, :, i, j]))


    # def _test_compute_weights_with_random_pad(self):
    #     """Compare weights under random padding mask."""
    #     for _ in tqdm(range(self.num_iterations), desc="Testing weights w/ pad mask"):
    #         batch_size = np.random.randint(1, 5)
    #         seq_length = np.random.randint(5, 15)

    #         x = self._generate_random_input(batch_size, seq_length)
    #         pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

    #         # Vectorized path
    #         with torch.no_grad():
    #             q_vec = self.vectorized_att.W_q(x).view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0,2,1,3)
    #             k_vec = self.vectorized_att.W_k(x).view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0,2,1,3)
    #             scores_vec = self.vectorized_att._key_query_product(q_vec, k_vec)

    #             causal = self.vectorized_att._default_mask(batch_size, seq_length)
    #             final_bool = self.vectorized_att.create_final_mask(causal, pad_mask)
    #             #   final_float = self.vectorized_att._process_final_mask(final_bool)

    #             weights_vec = self.vectorized_att._compute_weights(scores_vec, final_bool)

    #         # Naive path per head
    #         for h in range(self.num_heads):
    #             q_h = self.naive_att.W_q_list[h](x)
    #             k_h = self.naive_att.W_k_list[h](x)
    #             scores_h = self.naive_att._key_query_product(q_h, k_h)

    #             weights_h = self.naive_att._compute_weights(scores_h, final_bool[:, h])

    #             # Check values are approximately equal
    #             self.assertTrue(torch.allclose(weights_vec[:, h], weights_h, atol=5e-6))

    #         for b in range(batch_size):
    #             for i in range(seq_length):
    #                 for j in range(seq_length):
    #                     if j > i or not(pad_mask[b, i]) or not(pad_mask[b, j]):
    #                         self.assertTrue(torch.all(weights_vec[b, :, i, j] < 1e-9), msg="the weight should be 0 for masked positions")


    def _test_full_forward_with_pad(self):
        """Compare full forward outputs with pad mask."""
        for _ in tqdm(range(self.num_iterations), desc="Testing forward w/ pad mask"):
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)

            x = self._generate_random_input(batch_size, seq_length)
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

            with torch.no_grad():
                out_vec = self.vectorized_att(x, pad_mask)
                out_naive = self.naive_att(x, pad_mask)

            self.assertEqual(out_vec.shape, out_naive.shape)

            self.assertTrue(torch.allclose(out_vec, out_naive, atol=1e-6))


    # add the tests from the CustomModuleBaseTest
    def _test_eval_mode(self) -> None:
        """Test that calling eval() sets training=False for all parameters and submodules"""
        super()._test_eval_mode(self.vectorized_att)

    def _test_train_mode(self) -> None:
        """Test that calling train() sets training=True for all parameters and submodules"""
        super()._test_train_mode(self.vectorized_att)   

    def _test_consistent_output_without_dropout_bn(self) -> None:
        """
        Test that modules without dropout or batch normalization 
        produce consistent output for the same input
        """
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(10, 10)
            super()._test_consistent_output_without_dropout_bn(self.vectorized_att, input_tensor)

    def _test_consistent_output_in_eval_mode(self) -> None:
        """Test that all modules in eval mode produce consistent output for the same input"""
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(10, 10)
            super()._test_consistent_output_in_eval_mode(self.vectorized_att, input_tensor)

    def _test_batch_size_one_in_train_mode(self) -> None:
        """
        Test that modules with batch normalization layers might raise errors 
        with batch size 1 in train mode
        """
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(1, 10)
            super()._test_batch_size_one_in_train_mode(self.vectorized_att, input_tensor)


    def _test_batch_size_one_in_eval_mode(self) -> None:
        """Test that modules in eval mode should not raise errors for batch size 1"""
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(1, 10)
            super()._test_batch_size_one_in_eval_mode(self.vectorized_att, input_tensor) 

    def _test_named_parameters_length(self) -> None:
        """Test that named_parameters() and parameters() have the same length"""
        super()._test_named_parameters_length(self.vectorized_att)

    def _test_to_device(self) -> None:
        """Test that module can move between devices properly"""
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(10, 10)
            super()._test_to_device(self.vectorized_att, input_tensor) 

    @unittest.skip("passed")
    def _test_gradcheck(self) -> None:
        """Test gradient computation using torch.autograd.gradcheck."""
        # Use small dimensions for speed
        for _ in tqdm(range(100), desc="Testing gradcheck"):
            b, s = (np.random.randint(2, 10) for _ in range(2))
            input_tensor = self._generate_random_input(batch_size=b, seq_length=s)
            pad_mask = self._generate_random_pad_mask(batch_size=b, seq_length=s)
        
            # Test without mask
            super()._test_gradcheck(self.vectorized_att, input_tensor)
            # Test with mask
            super()._test_gradcheck(self.vectorized_att, input_tensor, pad_mask)

    @unittest.skip("passed")
    def _test_gradcheck_large_values(self) -> None:
        """Test gradient computation with large input values."""
        # Use small dimensions for speed
        for _ in tqdm(range(100), desc="Testing gradcheck with large values"):
            b, s = (np.random.randint(2, 10) for _ in range(2))
            input_tensor = self._generate_random_input(batch_size=b, seq_length=s)
            pad_mask = self._generate_random_pad_mask(batch_size=b, seq_length=s)
        
            # Test without mask
            super()._test_gradcheck_large_values(self.vectorized_att, input_tensor)
            # Test with mask
            super()._test_gradcheck_large_values(self.vectorized_att, input_tensor, pad_mask)
    
    @unittest.skip("passed")
    def _test_grad_against_nan(self) -> None:
        """Test that the gradient is not nan"""
        for _ in range(self.num_iterations):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = self._generate_random_input(batch_size, seq_length)
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)
            super()._test_grad_against_nan(self.vectorized_att, input_tensor, pad_mask)
            super()._test_grad_against_nan(self.vectorized_att, input_tensor)

    @unittest.skip("passed")
    def _test_grad_against_nan_large_values(self) -> None:
        """Test that the gradient is not nan with large input values"""
        for _ in range(self.num_iterations):
            batch_size = np.random.randint(1, 10)
            seq_length = np.random.randint(5, 20)
            input_tensor = self._generate_random_input(batch_size, seq_length)
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)
            super()._test_grad_against_nan_large_values(self.vectorized_att, input_tensor, pad_mask)
            super()._test_grad_against_nan_large_values(self.vectorized_att, input_tensor)


class TestCausalMultiHeadAttentionLayer(TestMultiHeadAttentionLayer):
    def setUp(self):
        # Define common dimensions for testing
        self.d_model = 32
        self.num_heads = 4
        self.key_dim = 8 
        self.value_dim = 8  
        
        # Number of test iterations
        self.num_iterations = 1000
        
        # Create both implementations
        self.vectorized_att = CausalMultiHeadAttentionLayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim
        )
        
        self.naive_att = CausalNaiveMHA(
            d_model=self.d_model,
            num_heads=self.num_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim
        )
        
        # Set both to the same random weights
        self._sync_weights() 


    def test_default_mask_creation(self):
        self._test_default_mask_creation()

    def test_query_key_product(self):
        self._test_query_key_product()


    def _test_compute_weights(self):
        """Test that both implementations compute the same attention weights"""
        for _ in tqdm(range(self.num_iterations), desc="Testing compute weights"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)
            
            # Generate random input
            x = self._generate_random_input(batch_size, seq_length)
            
            # Process through both implementations to get query-key products
            with torch.no_grad():
                # For vectorized implementation
                q_vec = self.vectorized_att.W_q(x)
                k_vec = self.vectorized_att.W_k(x)
                q_vec = q_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
                k_vec = k_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
                vec_product = self.vectorized_att._key_query_product(q_vec, k_vec)
                
                # Create mask
                vec_mask = self.vectorized_att._default_mask(batch_size, seq_length)
                vec_mask = self.vectorized_att.create_final_mask(vec_mask, None)
                # vec_mask = self.vectorized_att._process_final_mask(vec_mask)
                
                # For naive implementation, process each head separately
                q_naive_list = [None for _ in range(self.num_heads)]
                k_naive_list = [None for _ in range(self.num_heads)]
                for h in range(self.num_heads):
                    q_naive_list[h] = self.naive_att.W_q_list[h].forward(x)
                    k_naive_list[h] = self.naive_att.W_k_list[h].forward(x)
                
                naive_mask = self.naive_att._default_mask(batch_size, seq_length)
                naive_mask = self.naive_att.create_final_mask(naive_mask, None)
                # naive_mask = self.naive_att._process_final_mask (naive_mask)

            self.assertTrue(torch.equal(vec_mask, naive_mask))

            # Compute weights using vectorized implementation
            vec_weights = self.vectorized_att._compute_weights(vec_product, vec_mask)
            
            # Test for each head in the naive implementation
            for h in range(self.num_heads):
                naive_product = self.naive_att._key_query_product(q_naive_list[h], k_naive_list[h])
                naive_weights = self.naive_att._compute_weights(naive_product, naive_mask[:, h])
                
                # Check shape match for this head       
                self.assertEqual(vec_weights[:, h].shape, naive_weights.shape)

                # Check values are approximately equal
                validated = False
                for tol_coeff in range(1, 6):
                    try:
                        self.assertTrue(torch.allclose(vec_weights[:, h], naive_weights, atol=tol_coeff*1e-7)) # can tolerate up to 5e-7
                        validated = True
                        break
                    except:
                        continue

                self.assertTrue(validated, msg="Vectorised and naive weights diverge")

            # at this point we know both implementations produce the same weights
            # so we can test the weights for the masked positions
            for b in range(batch_size):
                for i in range(seq_length):
                    for j in range(seq_length):
                        if j > i:
                            self.assertTrue(torch.all(vec_weights[b, :, i, j] < 1e-9), msg="the weight should be 0 for masked positions")


    def _test_compute_weighted_values(self):
        self._test_compute_weighted_values()


    def test_full_forward_pass(self):
        self._test_full_forward_pass()
        
    def _test_create_final_mask_with_pad(self):
        """final_mask from both implementations agrees and respects causal+pad."""
        for _ in tqdm(range(self.num_iterations), desc="Testing final mask creation w/ pad"):
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)

            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

            causal_vec = self.vectorized_att._default_mask(batch_size, seq_length)
            causal_naive = self.naive_att._default_mask(batch_size, seq_length)

            vec_final = self.vectorized_att.create_final_mask(causal_vec, pad_mask)
            naive_final = self.naive_att.create_final_mask(causal_naive, pad_mask)

            self.assertTrue(torch.equal(vec_final, naive_final))

            # verify causal+pad rules
            for b in range(batch_size):
                for i in range(seq_length):
                    for j in range(seq_length):
                        if j > i or not pad_mask[b, i] or not pad_mask[b, j]:
                            self.assertFalse(torch.any(vec_final[b, :, i, j]))
                        else:
                            self.assertTrue(torch.all(vec_final[b, :, i, j]))


    def _test_compute_weights_with_random_pad(self):
        """Compare weights under random padding mask."""
        for _ in tqdm(range(self.num_iterations), desc="Testing weights w/ pad mask"):
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)

            x = self._generate_random_input(batch_size, seq_length)
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

            # Vectorized path
            with torch.no_grad():
                q_vec = self.vectorized_att.W_q(x).view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0,2,1,3)
                k_vec = self.vectorized_att.W_k(x).view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0,2,1,3)
                scores_vec = self.vectorized_att._key_query_product(q_vec, k_vec)

                causal = self.vectorized_att._default_mask(batch_size, seq_length)
                final_bool = self.vectorized_att.create_final_mask(causal, pad_mask)
                #   final_float = self.vectorized_att._process_final_mask(final_bool)

                weights_vec = self.vectorized_att._compute_weights(scores_vec, final_bool)

            # Naive path per head
            for h in range(self.num_heads):
                q_h = self.naive_att.W_q_list[h](x)
                k_h = self.naive_att.W_k_list[h](x)
                scores_h = self.naive_att._key_query_product(q_h, k_h)

                weights_h = self.naive_att._compute_weights(scores_h, final_bool[:, h])

                # Check values are approximately equal
                self.assertTrue(torch.allclose(weights_vec[:, h], weights_h, atol=5e-6))

            for b in range(batch_size):
                for i in range(seq_length):
                    for j in range(seq_length):
                        if j > i or not(pad_mask[b, i]) or not(pad_mask[b, j]):
                            self.assertTrue(torch.all(weights_vec[b, :, i, j] < 1e-9), msg="the weight should be 0 for masked positions")

    def test_full_forward_with_pad(self):
        self._test_full_forward_with_pad()  

    def test_eval_mode(self):
        self._test_eval_mode()

    def test_train_mode(self):
        self._test_train_mode()


    def test_consistent_output_without_dropout_bn(self):
        self._test_consistent_output_without_dropout_bn()

    def test_consistent_output_in_eval_mode(self):
        self._test_consistent_output_in_eval_mode()

    def test_batch_size_one_in_train_mode(self):
        self._test_batch_size_one_in_train_mode()

    def test_batch_size_one_in_eval_mode(self):
        self._test_batch_size_one_in_eval_mode()

    def test_named_parameters_length(self): 
        self._test_named_parameters_length()

    def test_to_device(self):
        self._test_to_device()

    def test_gradcheck(self):
        self._test_gradcheck()  

    def test_gradcheck_large_values(self):
        self._test_gradcheck_large_values()

    def test_grad_against_nan(self):
        self._test_grad_against_nan()   

    def test_grad_against_nan_large_values(self):
        self._test_grad_against_nan_large_values()


class TestBidirectionalMultiHeadAttentionLayer(TestMultiHeadAttentionLayer):

    def setUp(self):
        self.d_model = 32
        self.num_heads = 4
        self.key_dim = 8 
        self.value_dim = 8

        self.num_iterations = 1000
        
        self.vectorized_att = BidirectionalMultiHeadAttentionLayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim    
        )
        
        self.naive_att = BidirectionalNaiveMHA(
            d_model=self.d_model,
            num_heads=self.num_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim
        )

        self._sync_weights() 


    def test_default_mask_creation(self):
        self._test_default_mask_creation()

    def test_query_key_product(self):
        self._test_query_key_product()

    def _test_compute_weights(self):
        """Test that both implementations compute the same attention weights"""
        for _ in tqdm(range(self.num_iterations), desc="Testing compute weights"):
            # Generate random batch size and sequence length
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)
            
            # Generate random input
            x = self._generate_random_input(batch_size, seq_length)
            
            # Process through both implementations to get query-key products
            with torch.no_grad():
                # For vectorized implementation
                q_vec = self.vectorized_att.W_q(x)
                k_vec = self.vectorized_att.W_k(x)
                q_vec = q_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
                k_vec = k_vec.view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
                vec_product = self.vectorized_att._key_query_product(q_vec, k_vec)
                
                # Create mask
                vec_mask = self.vectorized_att._default_mask(batch_size, seq_length)
                vec_mask = self.vectorized_att.create_final_mask(vec_mask, None)
                # vec_mask = self.vectorized_att._process_final_mask(vec_mask)
                
                # For naive implementation, process each head separately
                q_naive_list = [None for _ in range(self.num_heads)]
                k_naive_list = [None for _ in range(self.num_heads)]
                for h in range(self.num_heads):
                    q_naive_list[h] = self.naive_att.W_q_list[h].forward(x)
                    k_naive_list[h] = self.naive_att.W_k_list[h].forward(x)
                
                naive_mask = self.naive_att._default_mask(batch_size, seq_length)
                naive_mask = self.naive_att.create_final_mask(naive_mask, None)
                # naive_mask = self.naive_att._process_final_mask (naive_mask)

            self.assertTrue(torch.equal(vec_mask, naive_mask))

            # Compute weights using vectorized implementation
            vec_weights = self.vectorized_att._compute_weights(vec_product, vec_mask)
            
            # Test for each head in the naive implementation
            for h in range(self.num_heads):
                naive_product = self.naive_att._key_query_product(q_naive_list[h], k_naive_list[h])
                naive_weights = self.naive_att._compute_weights(naive_product, naive_mask[:, h])
                
                # Check shape match for this head       
                self.assertEqual(vec_weights[:, h].shape, naive_weights.shape)

                # Check values are approximately equal
                validated = False
                for tol_coeff in range(1, 6):
                    try:
                        self.assertTrue(torch.allclose(vec_weights[:, h], naive_weights, atol=tol_coeff*1e-7)) # can tolerate up to 5e-7
                        validated = True
                        break
                    except:
                        continue

                self.assertTrue(validated, msg="Vectorised and naive weights diverge")


    def test_compute_weighted_values(self):
        self._test_compute_weighted_values()

    def test_full_forward_pass(self):
        self._test_full_forward_pass()

    def _test_create_final_mask_with_pad(self):
        """final_mask from both implementations agrees and respects causal+pad."""
        for _ in tqdm(range(self.num_iterations), desc="Testing final mask creation w/ pad"):
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)

            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

            causal_vec = self.vectorized_att._default_mask(batch_size, seq_length)
            causal_naive = self.naive_att._default_mask(batch_size, seq_length)

            vec_final = self.vectorized_att.create_final_mask(causal_vec, pad_mask)
            naive_final = self.naive_att.create_final_mask(causal_naive, pad_mask)

            self.assertTrue(torch.equal(vec_final, naive_final))

            # verify causal+pad rules
            for b in range(batch_size):
                for i in range(seq_length):
                    for j in range(seq_length):
                        if not pad_mask[b, i] or not pad_mask[b, j]:
                            self.assertFalse(torch.any(vec_final[b, :, i, j]))
                        else:
                            self.assertTrue(torch.all(vec_final[b, :, i, j]))


    def _test_compute_weights_with_random_pad(self):
        """Compare weights under random padding mask."""
        for _ in tqdm(range(self.num_iterations), desc="Testing weights w/ pad mask"):
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(5, 15)

            x = self._generate_random_input(batch_size, seq_length)
            pad_mask = self._generate_random_pad_mask(batch_size, seq_length)

            # Vectorized path
            with torch.no_grad():
                q_vec = self.vectorized_att.W_q(x).view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0,2,1,3)
                k_vec = self.vectorized_att.W_k(x).view(batch_size, seq_length, self.num_heads, self.key_dim).permute(0,2,1,3)
                scores_vec = self.vectorized_att._key_query_product(q_vec, k_vec)

                causal = self.vectorized_att._default_mask(batch_size, seq_length)
                final_bool = self.vectorized_att.create_final_mask(causal, pad_mask)
                #   final_float = self.vectorized_att._process_final_mask(final_bool)

                weights_vec = self.vectorized_att._compute_weights(scores_vec, final_bool)

            # Naive path per head
            for h in range(self.num_heads):
                q_h = self.naive_att.W_q_list[h](x)
                k_h = self.naive_att.W_k_list[h](x)
                scores_h = self.naive_att._key_query_product(q_h, k_h)

                weights_h = self.naive_att._compute_weights(scores_h, final_bool[:, h])

                # Check values are approximately equal
                self.assertTrue(torch.allclose(weights_vec[:, h], weights_h, atol=5e-6))

            for b in range(batch_size):
                for i in range(seq_length):
                    for j in range(seq_length):
                        if not(pad_mask[b, i]) or not(pad_mask[b, j]):
                            self.assertTrue(torch.all(weights_vec[b, :, i, j] < 1e-9), msg="the weight should be 0 for masked positions")

    def test_full_forward_with_pad(self):
        self._test_full_forward_with_pad()  

    def test_eval_mode(self):
        self._test_eval_mode()

    def test_train_mode(self):
        self._test_train_mode()

    def test_consistent_output_without_dropout_bn(self):
        self._test_consistent_output_without_dropout_bn()

    def test_consistent_output_in_eval_mode(self):
        self._test_consistent_output_in_eval_mode()

    def test_batch_size_one_in_train_mode(self):    
        self._test_batch_size_one_in_train_mode()

    def test_batch_size_one_in_eval_mode(self):
        self._test_batch_size_one_in_eval_mode()

    def test_named_parameters_length(self):     
        self._test_named_parameters_length()

    def test_to_device(self):
        self._test_to_device()


    @unittest.skip("passed")
    def test_gradcheck(self):       
        self._test_gradcheck()

    @unittest.skip("passed")
    def test_gradcheck_large_values(self):
        self._test_gradcheck_large_values()

    @unittest.skip("passed")
    def test_grad_against_nan(self):
        self._test_grad_against_nan()

    @unittest.skip("passed")
    def test_grad_against_nan_large_values(self):
        self._test_grad_against_nan_large_values()


if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main() 