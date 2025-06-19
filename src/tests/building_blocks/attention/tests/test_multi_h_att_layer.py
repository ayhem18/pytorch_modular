import torch
import unittest
import numpy as np

from tqdm import tqdm

from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.attention.multi_head_att import MultiHeadAttentionLayer
from tests.building_blocks.attention.naive_implementations.multi_head_att import NaiveMHA


class TestMultiHeadAttentionLayer(CustomModuleBaseTest):
    def setUp(self):
        # Define common dimensions for testing
        self.d_model = 32
        self.num_heads = 4
        self.key_dim = 8 
        self.value_dim = 8  
        
        # Number of test iterations
        self.num_iterations = 1000
        
        # Create both implementations
        self.vectorized_att = MultiHeadAttentionLayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim
        )
        
        self.naive_att = NaiveMHA(
            d_model=self.d_model,
            num_heads=self.num_heads,
            value_dim=self.value_dim,
            key_dim=self.key_dim
        )
        
        # Set both to the same random weights
        self._sync_weights()
        
    def _sync_weights(self):
        """Synchronize weights between vectorized and naive implementations"""
        self.naive_att.sync_weights_from_vectorized(self.vectorized_att)
    
    def _generate_random_input(self, batch_size: int, seq_length: int) -> torch.Tensor:
        """Generate random input tensor"""
        return torch.randn(batch_size, seq_length, self.d_model)
    
    def test_attention_mask_creation(self):
        """Test that both implementations create the same attention mask"""
        for _ in tqdm(range(self.num_iterations), desc="Testing attention mask creation"):
            # Generate a random sequence length between 5 and 20
            seq_length = np.random.randint(5, 20)
            
            # Generate masks from both implementations
            vec_mask = self.vectorized_att._create_attention_mask(seq_length)
            naive_mask = self.naive_att._create_attention_mask(seq_length)
            
            # Check they have the same shape
            self.assertEqual(vec_mask.shape, naive_mask.shape)
            
            # Check that they are equal (accounting for floating point precision)
            validated = False
            
            for tol_coeff in range(1, 6):
                try:
                    self.assertTrue(torch.allclose(vec_mask, naive_mask, atol=tol_coeff*1e-7))  
                    validated = True
                    break
                except:
                    continue

            if not validated:
                self.fail(f"No tolerance coefficient found for which the values are approximately equal") 

    def test_query_key_product(self):
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

                if not validated:
                    self.fail(f"No tolerance coefficient found for which the values are approximately equal")
    
    def test_compute_weights(self):
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
                vec_mask = self.vectorized_att._create_attention_mask(seq_length)
                vec_mask = vec_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
                
                # For naive implementation, process each head separately
                q_naive_list = [None for _ in range(self.num_heads)]
                k_naive_list = [None for _ in range(self.num_heads)]
                for h in range(self.num_heads):
                    q_naive_list[h] = self.naive_att.W_q_list[h].forward(x)
                    k_naive_list[h] = self.naive_att.W_k_list[h].forward(x)
                
                naive_mask = self.naive_att._create_attention_mask(seq_length)
            
            # Compute weights using vectorized implementation
            vec_weights = self.vectorized_att._compute_weights(vec_product, vec_mask)
            
            # Test for each head in the naive implementation
            for h in range(self.num_heads):
                naive_product = self.naive_att._key_query_product(q_naive_list[h], k_naive_list[h])
                naive_weights = self.naive_att._compute_weights(naive_product, naive_mask)
                
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

                if not validated:
                    self.fail(f"No tolerance coefficient found for which the values are approximately equal")
    

    def test_compute_weighted_values(self):
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
                
                vec_mask = self.vectorized_att._create_attention_mask(seq_length)
                vec_mask = vec_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
                
                vec_weights = self.vectorized_att._compute_weights(vec_product, vec_mask)
                
                # For naive implementation, process each head separately
                q_naive_list = [None for _ in range(self.num_heads)]
                k_naive_list = [None for _ in range(self.num_heads)]
                v_naive_list = [None for _ in range(self.num_heads)]

                for h in range(self.num_heads):
                    q_naive_list[h] = self.naive_att.W_q_list[h].forward(x)
                    k_naive_list[h] = self.naive_att.W_k_list[h].forward(x)
                    v_naive_list[h] = self.naive_att.W_v_list[h].forward(x)
                
                naive_mask = self.naive_att._create_attention_mask(seq_length)
            
            # Compute weighted values using vectorized implementation
            vec_output = self.vectorized_att._compute_new_v(vec_weights, v_vec)
            
            # Test for each head in the naive implementation
            for h in range(self.num_heads):
                naive_product = self.naive_att._key_query_product(q_naive_list[h], k_naive_list[h])
                naive_weights = self.naive_att._compute_weights(naive_product, naive_mask)
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

                if not validated:
                    self.fail(f"No tolerance coefficient found for which the values are approximately equal")

    def test_full_forward_pass(self):
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

            if not validated:
                self.fail(f"No tolerance coefficient found for which the values are approximately equal")


    # add the tests from the CustomModuleBaseTest
    def test_eval_mode(self) -> None:
        """Test that calling eval() sets training=False for all parameters and submodules"""
        super()._test_eval_mode(self.vectorized_att)

    def test_train_mode(self) -> None:
        """Test that calling train() sets training=True for all parameters and submodules"""
        super()._test_train_mode(self.vectorized_att)   

    def test_consistent_output_without_dropout_bn(self) -> None:
        """
        Test that modules without dropout or batch normalization 
        produce consistent output for the same input
        """
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(10, 10)
            super()._test_consistent_output_without_dropout_bn(self.vectorized_att, input_tensor)

    def test_consistent_output_in_eval_mode(self) -> None:
        """Test that all modules in eval mode produce consistent output for the same input"""
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(10, 10)
            super()._test_consistent_output_in_eval_mode(self.vectorized_att, input_tensor)

    def test_batch_size_one_in_train_mode(self) -> None:
        """
        Test that modules with batch normalization layers might raise errors 
        with batch size 1 in train mode
        """
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(1, 10)
            super()._test_batch_size_one_in_train_mode(self.vectorized_att, input_tensor)


    def test_batch_size_one_in_eval_mode(self) -> None:
        """Test that modules in eval mode should not raise errors for batch size 1"""
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(1, 10)
            super()._test_batch_size_one_in_eval_mode(self.vectorized_att, input_tensor) 

    def test_named_parameters_length(self) -> None:
        """Test that named_parameters() and parameters() have the same length"""
        super()._test_named_parameters_length(self.vectorized_att)

    def test_to_device(self) -> None:
        """Test that module can move between devices properly"""
        for _ in range(self.num_iterations):
            input_tensor = self._generate_random_input(10, 10)
            super()._test_to_device(self.vectorized_att, input_tensor) 

    

if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main() 