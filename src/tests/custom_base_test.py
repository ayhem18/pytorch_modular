import torch
import unittest
from torch.autograd import gradcheck
from typing import Any, List, Tuple, Union


class CustomModuleBaseTest(unittest.TestCase):
    """
    Base test class for custom PyTorch modules.
    Provides common test methods that should apply to all properly-implemented modules.
    """
    
    def _test_module_is_nn_module(self, block: torch.nn.Module) -> None:
        """Test that the module is an instance of torch.nn.Module. THIS IS VERY IMPORTANT"""
        self.assertIsInstance(block, torch.nn.Module, "Module should be an instance of torch.nn.Module")

    # --- helper methods ---

    def _get_valid_input(self, *args, **kwargs) -> torch.Tensor:
        """
        Generate a random input tensor with the correct shape.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _has_stochastic_layers(self, block: torch.nn.Module) -> bool:
        """Check if the module contains dropout or batch normalization layers"""
        for module in block.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.RMSNorm)):
                return True
        return False


    def _test_outputs(self, output1: Union[torch.Tensor, Tuple, List], output2: Union[torch.Tensor, Tuple, List]) -> None:
        """Test that outputs are identical"""

        if isinstance(output1, torch.Tensor):
            self.assertIsInstance(output2, torch.Tensor, "Output should be a tensor")
            self.assertTrue(torch.allclose(output1, output2),
                         "Module in eval mode should produce consistent outputs")
            return 
        
        if isinstance(output1, (tuple, list)):
            self.assertIsInstance(output2, (tuple, list), "Output should be a tuple or list")
            self.assertEqual(len(output1), len(output2), "Output tuples or lists should have the same length")

            for o1, o2 in zip(output1, output2):
                self._test_outputs(o1, o2)
                
        else:
            self.fail("Output should be a tensor, a tuple or a list")


    def _set_arguments_to_device(self, arg: Union[torch.Tensor, Tuple, List, Any], device: str) -> Union[torch.Tensor, Tuple, List]:
        """Set the arguments to CUDA"""
        if isinstance(arg, torch.Tensor):
            return arg.to(device)
        elif isinstance(arg, (tuple, list)):
            return [self._set_arguments_to_device(a, device) for a in arg]
        return arg


    # ---- tests ----

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
        
        self._test_outputs(output1, output2)
    
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
            _ = block.forward(input_tensor, *args, **kwargs)
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
            args = [self._set_arguments_to_device(arg, 'cuda') for arg in args]

        if len(kwargs) > 0:
            kwargs = {k: self._set_arguments_to_device(v, 'cuda') for k, v in kwargs.items()}

        # Move to CUDA
        cuda_block = block.to('cuda')
        
        # Check that device transfer returns the module itself
        self.assertIs(cuda_block, block, "to() method should return self")
        
        # Check that all parameters are moved to CUDA
        for param in cuda_block.parameters():
            self.assertTrue(param.is_cuda, "All parameters should be moved to CUDA")
        
        # move the input tensor to the device
        input_tensor = self._set_arguments_to_device(input_tensor, 'cuda')

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
            args = [self._set_arguments_to_device(arg, 'cpu') for arg in args]

        if len(kwargs) > 0:
            kwargs = {k: self._set_arguments_to_device(v, 'cpu') for k, v in kwargs.items()}

        # move the input tensor to the cpu
        input_tensor = self._set_arguments_to_device(input_tensor, 'cpu')

        try:
            block = block.to('cpu').eval()
            _ = block(input_tensor, *args, **kwargs)
        except Exception as e:
            self.fail(f"Calling the module on a tensor moved to cpu should not raise an error. Got: {e}")


    def _test_gradcheck(self, block: torch.nn.Module, input_tensor: torch.Tensor, *args, **kwargs) -> None:
        """Test gradient computation using torch.autograd.gradcheck."""
        # gradcheck needs double precision and the block to be in eval mode for reproducibility
        block_double = block.to(torch.double).eval()
        input_tensor_double = input_tensor.to(torch.double)
        
        # Convert args and kwargs to double as well if they are tensors
        args_double = [arg.to(torch.double) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs_double = {k: v.to(torch.double) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        input_tensor_double.requires_grad = True
        
        # gradcheck expects a tuple of inputs. We are only checking gradients wrt input_tensor.
        test_passed = gradcheck(lambda inp: block_double(inp, *args_double, **kwargs_double), (input_tensor_double,), atol=1e-8)
        self.assertTrue(test_passed, "gradcheck failed for the module.")


    def _test_gradcheck_large_values(self, block: torch.nn.Module, input_tensor: torch.Tensor, *args, **kwargs) -> None:
        """Test gradient computation with large input values."""
        # gradcheck needs double precision and the block to be in eval mode for reproducibility
        block_double = block.to(torch.double).eval()
        input_tensor_double = input_tensor.to(torch.double)

        # Convert args and kwargs to double as well if they are tensors
        args_double = [arg.to(torch.double) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs_double = {k: v.to(torch.double) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        input_tensor_double = input_tensor_double * 1e4
        input_tensor_double.requires_grad = True

        # Using a higher tolerance for large values as gradients can be very large.
        test_passed = gradcheck(lambda inp: block_double(inp, *args_double, **kwargs_double), (input_tensor_double,))
        self.assertTrue(test_passed, "gradcheck failed for the module with large input values.")


    def _test_grad_against_nan(self, block: torch.nn.Module, input_tensor: torch.Tensor, *args, **kwargs) -> None:
        """Test that the gradient is not nan"""
        block_double = block.to(torch.double).train()
        input_tensor_double = input_tensor.to(torch.double)
        input_tensor_double.requires_grad = True


        args_double = [arg.to(torch.double) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs_double = {k: v.to(torch.double) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        output = block_double(input_tensor_double, *args_double, **kwargs_double)

        output.sum().backward()

        self.assertFalse(torch.isnan(input_tensor_double.grad).any(), "Nan values in the gradient")

        for param in block_double.parameters():
            self.assertFalse(torch.isnan(param.grad).any(), "Nan values in the gradient")
        

    def _test_grad_against_nan_large_values(self, block: torch.nn.Module, input_tensor: torch.Tensor, *args, **kwargs) -> None:
        """Test that the gradient is not nan"""
        block_double = block.to(torch.double).eval()
        input_tensor_double = input_tensor.to(torch.double)

        input_tensor_double = input_tensor_double * 1e4
        input_tensor_double.requires_grad = True

        args_double = [arg.to(torch.double) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs_double = {k: v.to(torch.double) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        output = block_double(input_tensor_double, *args_double, **kwargs_double)

        output.sum().backward()

        self.assertFalse(torch.isnan(input_tensor_double.grad).any(), "Nan values in the gradient with large input values") 

        for param in block_double.parameters():
            self.assertFalse(torch.isnan(param.grad).any(), "Nan values in the gradient with large input values")