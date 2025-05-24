import torch
import unittest


class CustomModuleBaseTest(unittest.TestCase):
    """
    Base test class for custom PyTorch modules.
    Provides common test methods that should apply to all properly-implemented modules.
    """
    
    def _get_valid_input(self, *args, **kwargs) -> torch.Tensor:
        """
        Generate a random input tensor with the correct shape.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _test_eval_mode(self, block: torch.nn.Module) -> None:
        """Test that calling eval() sets training=False for all parameters and submodules"""
        block.eval()
        
        # Check that the module itself is in eval mode
        self.assertFalse(block.training, "Module should be in eval mode after calling eval()")
        
        # Check all child modules
        for module in block.modules():
            self.assertFalse(module.training, f"Submodule {module.__class__.__name__} should be in eval mode")
    
    def _test_train_mode(self, block: torch.nn.Module) -> None:
        """Test that calling train() sets training=True for all parameters and submodules"""
        block.train()
        
        # Check that the module itself is in training mode
        self.assertTrue(block.training, "Module should be in training mode after calling train()")
        
        # Check all child modules
        for module in block.modules():
            self.assertTrue(module.training, f"Submodule {module.__class__.__name__} should be in training mode")
    
    def _has_stochastic_layers(self, block: torch.nn.Module) -> bool:
        """Check if the module contains dropout or batch normalization layers"""
        for module in block.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.RMSNorm)):
                return True
        return False
    
    def _test_consistent_output_without_dropout_bn(self, block: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """
        Test that modules without dropout or batch normalization 
        produce consistent output for the same input
        """
        if self._has_stochastic_layers(block):
            return  # Skip test if the module has stochastic layers
        
        block.train()
        
        # Get outputs from multiple forward passes
        output1 = block(input_tensor)
        output2 = block(input_tensor)
        
        # Check that outputs are identical
        self.assertTrue(torch.allclose(output1, output2),
                         "Module without stochastic layers should produce consistent outputs")
    
    def _test_consistent_output_in_eval_mode(self, block: torch.nn.Module, input_tensor: torch.Tensor, *args, **kwargs) -> None:
        """Test that all modules in eval mode produce consistent output for the same input"""
        # Set to eval mode
        block.eval()
        
        # Get outputs from multiple forward passes
        output1 = block(input_tensor, *args, **kwargs)
        output2 = block(input_tensor, *args, **kwargs)
        
        # Check that outputs are identical
        self.assertTrue(torch.allclose(output1, output2),
                         "Module in eval mode should produce consistent outputs")
    
    def _test_batch_size_one_in_train_mode(self, block: torch.nn.Module, input_tensor: torch.Tensor, *args, **kwargs) -> None:
        """
        Test that modules with batch normalization layers might raise errors 
        with batch size 1 in train mode
        """
        if input_tensor.size(0) != 1:
            raise ValueError("Input tensor should have batch size 1")

        # First check if the module has batch normalization layers
        has_batchnorm = False
        for module in block.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                has_batchnorm = True
                break
        
        if not has_batchnorm:
            return  # Skip test if no batch normalization layers
        
        # Set to train mode
        block.train()
        
        
        # In train mode, BatchNorm1d with batch size 1 might raise an error
        # since variance can't be computed properly
        try:
            _ = block(input_tensor, *args, **kwargs)
            # If it doesn't raise an error, it's acceptable
        except Exception as e:
            # Check if the error is related to batch size and BatchNorm
            self.assertIn('Expected more than 1 value per channel when training', str(e), 
                          "Module with BatchNorm in train mode may raise errors with batch size 1")

    def _test_batch_size_one_in_eval_mode(self, block: torch.nn.Module, input_tensor: torch.Tensor, *args, **kwargs) -> None:
        """Test that modules in eval mode should not raise errors for batch size 1"""
        # Set to eval mode
        block.eval()
        
        if input_tensor.size(0) != 1:
            raise ValueError("Input tensor should have batch size 1")

        # This should not raise an error
        try:
            _ = block(input_tensor, *args, **kwargs)
        except Exception as e:
            self.fail(f"Module in eval mode should not raise errors with batch size 1. Got: {e}")
    
    def _test_named_parameters_length(self, block: torch.nn.Module) -> None:
        """Test that named_parameters() and parameters() have the same length"""
        named_params = list(block.named_parameters())
        params = list(block.parameters())
        
        self.assertEqual(len(named_params), len(params),
                         "named_parameters() and parameters() should have the same length")
        
        # Additional check: make sure each parameter has a unique name
        param_names = [name for name, _ in named_params]
        self.assertEqual(len(param_names), len(set(param_names)),
                         "All parameters should have unique names")
    
    def _test_to_device(self, block: torch.nn.Module, input_tensor: torch.Tensor, *args, **kwargs) -> None:
        """Test that module can move between devices properly"""
        # Only run if CUDA is available
        if not torch.cuda.is_available():
            return
        
        if len(args) > 0:
            args = [arg.to('cuda') if isinstance(arg, torch.Tensor) else arg for arg in args]

        if len(kwargs) > 0:
            kwargs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        # Move to CUDA
        cuda_block = block.to('cuda')
        
        # Check that device transfer returns the module itself
        self.assertIs(cuda_block, block, "to() method should return self")
        
        # Check that all parameters are moved to CUDA
        for param in cuda_block.parameters():
            self.assertTrue(param.is_cuda, "All parameters should be moved to CUDA")
        
        # move the input tensor to the device
        input_tensor = input_tensor.to('cuda')

        try:
            block = block.to('cuda').eval()
            _ =  block(input_tensor, *args, **kwargs)
        except Exception as e:
            self.fail(f"Calling the module on a tensor moved to cuda should not raise an error. Got: {e}")

        # Move back to CPU and check
        cpu_block = cuda_block.to('cpu')
        for param in cpu_block.parameters():
            self.assertFalse(param.is_cuda, "All parameters should be moved back to CPU")

        if len(args) > 0:
            args = [arg.to('cpu') if isinstance(arg, torch.Tensor) else arg for arg in args]

        if len(kwargs) > 0:
            kwargs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        # move the input tensor to the cpu
        input_tensor = input_tensor.to('cpu')

        try:
            block = block.to('cpu').eval()
            _ = block(input_tensor, *args, **kwargs)
        except Exception as e:
            self.fail(f"Calling the module on a tensor moved to cpu should not raise an error. Got: {e}")


    # TODO: better understand the clone behavior
    def _test_clone_method(self, block: torch.nn.Module) -> None:
        """Test the clone() method if it exists"""
        if not hasattr(block, 'clone'):
            return  # Skip if clone method doesn't exist
        
        # Clone the module
        cloned_block = block.clone()
        
        # Check that it's a different object
        self.assertIsNot(cloned_block, block, "clone() should return a new object")
        
        # Check that they have the same class
        self.assertIs(cloned_block.__class__, block.__class__, 
                     "clone() should return an object of the same class")
        
        # Check that they have the same number of parameters
        orig_params = list(block.parameters())
        cloned_params = list(cloned_block.parameters())
        self.assertEqual(len(orig_params), len(cloned_params),
                        "Original and cloned modules should have the same number of parameters")
        
        # Check that parameters have the same shape but are different objects
        for p1, p2 in zip(orig_params, cloned_params):
            self.assertEqual(p1.shape, p2.shape, 
                            "Corresponding parameters should have the same shape")
            self.assertIsNot(p1, p2, "Parameters should be different objects")
            # Values should be initially the same
            self.assertTrue(torch.allclose(p1, p2),
                           "Parameters should have the same values after cloning")
    