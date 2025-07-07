import torch
import random
import unittest

from typing import Optional, Tuple

from tests.custom_base_test import CustomModuleBaseTest
from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet1d import DiffusionUNetOneDim

class TestDiffusionUNetOneDim(CustomModuleBaseTest):
    """Test class for DiffusionUNetOneDim with various configuration combinations"""
    
    def setUp(self):
        """Initialize test parameters"""
        # Define common test parameters
        self.input_channels_range = (1, 3)
        self.output_channels_range = (1, 3)
        self.cond_dimension_range = (16, 64)
        self.num_classes_options = [None, 5, 10]
        self.embedding_methods = ['positional', 'gaussian_fourier']
        
        # For UNet blocks (at least 2 resnet blocks per layer.)
        self.num_resnet_blocks_range = (2, 4)
    
    def _get_valid_input(self, 
                        model: DiffusionUNetOneDim,
                        batch_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Generate input tensors appropriate for the model configuration"""

        num_layers = model.unet.num_down_layers
        height, weight = 2 ** (num_layers) * random.randint(4, 16), 2 ** (num_layers) * random.randint(4, 16)

        x = torch.randn(batch_size, model.input_channels, height, weight)
        time_step = torch.randint(0, 1000, (batch_size,))
        
        class_tensor = None
        if model.num_classes is not None:
            class_tensor = torch.randint(0, model.num_classes - 1, (batch_size,))
        
        return x, time_step, class_tensor
    
    def _build_unet_components(self, model: DiffusionUNetOneDim):
        """Build all components of the UNet model"""
        num_down_layers = random.randint(1, 2)
        num_res_blocks = random.randint(*self.num_resnet_blocks_range)
        
        down_params = {
            'num_down_layers': num_down_layers,
            'num_resnet_blocks': num_res_blocks,
            'out_channels': [32 * (2**i) for i in range(num_down_layers)],
            'downsample_types': 'conv',
            'inner_dim': 32,
            'dropout_rate': 0.0,
            'film_activation': 'relu',
            'force_residual': True
        }
        
        mid_params = {
            'num_resnet_blocks': num_res_blocks,
            'inner_dim': 32,
            'dropout_rate': 0.0,
            'film_activation': 'relu',
            'force_residual': True
        }
        
        up_params = {
            'num_resnet_blocks': num_res_blocks,
            'upsample_types': 'transpose_conv',
            'inner_dim': 32,
            'dropout_rate': 0.0,
            'film_activation': 'relu',
            'force_residual': True
        }
        
        model.unet.build_down_block(**down_params)
        model.unet.build_middle_block(**mid_params)
        model.unet.build_up_block(**up_params)
        
        return model
    
    def _generate_diffusion_unet_1d(self,
                                   input_channels=None,
                                   output_channels=None,
                                   cond_dimension=None,
                                   embedding_encoding_method=None,
                                   num_classes=None) -> DiffusionUNetOneDim:
        """Generate a DiffusionUNetOneDim with specified or random parameters"""
        if input_channels is None:
            input_channels = random.randint(*self.input_channels_range)
        
        if output_channels is None:
            output_channels = random.randint(*self.output_channels_range)
        
        if cond_dimension is None:
            cond_dimension = random.randint(*self.cond_dimension_range)
        
        if embedding_encoding_method is None:
            embedding_encoding_method = random.choice(self.embedding_methods)

        if num_classes is None:
            num_classes = random.choice(self.num_classes_options)
        
        if num_classes == False:
            num_classes = None

        model = DiffusionUNetOneDim(
            input_channels=input_channels,
            output_channels=output_channels,
            cond_dimension=cond_dimension,
            embedding_encoding_method=embedding_encoding_method,
            num_classes=num_classes
        )
        
        return model
    
    def test_initialization_without_class(self):
        """Test initialization without class conditioning"""
        for encoding in self.embedding_methods:
            model = self._generate_diffusion_unet_1d(
                embedding_encoding_method=encoding,
                num_classes=False
            )
            self.assertIsNotNone(model.conditions_processor.embedding_encoding)
            self.assertIsNotNone(model.conditions_processor.embedding_2d_projection)
            self.assertIsNone(model.conditions_processor.class_embedding)

    def test_initialization_with_class(self):
        """Test initialization with class conditioning"""
        for encoding in self.embedding_methods:
            num_classes = random.randint(5, 20)
            model = self._generate_diffusion_unet_1d(
                embedding_encoding_method=encoding,
                num_classes=num_classes
            )
            self.assertIsNotNone(model.conditions_processor.embedding_encoding)
            self.assertIsNotNone(model.conditions_processor.embedding_2d_projection)
            self.assertIsNotNone(model.conditions_processor.class_embedding)
            self.assertEqual(model.conditions_processor.class_embedding.num_embeddings, num_classes)
            
    def test_forward_without_class(self):
        """Test forward pass without class conditioning"""
        model = self._generate_diffusion_unet_1d(num_classes=False)
        self._build_unet_components(model)
        
        x, time_step, _ = self._get_valid_input(model)
        
        output = model(x, time_step)
        
        self.assertEqual(output.shape[0], x.shape[0])
        self.assertEqual(output.shape[1], model.output_channels)
        self.assertEqual(output.shape[2:], x.shape[2:])

    def test_forward_with_class(self):
        """Test forward pass with class conditioning"""
        num_classes = random.randint(5, 20)
        model = self._generate_diffusion_unet_1d(num_classes=num_classes)
        self._build_unet_components(model)
        
        x, time_step, class_tensor = self._get_valid_input(model)
        
        if class_tensor is not None:
            output = model(x, time_step, class_tensor)
        else:
            output = model(x, time_step)
        
        self.assertEqual(output.shape[0], x.shape[0])
        self.assertEqual(output.shape[1], model.output_channels)
        self.assertEqual(output.shape[2:], x.shape[2:])


    def test_eval_mode(self):
        """Test that the model can be set to evaluation mode"""
        models = [
            self._generate_diffusion_unet_1d(num_classes=None),
            self._generate_diffusion_unet_1d(num_classes=10),
        ]
        
        for model in models:
            self._build_unet_components(model)
            super()._test_eval_mode(model)
    
    def test_train_mode(self):
        """Test that the model can be set to training mode"""
        models = [
            self._generate_diffusion_unet_1d(num_classes=None),
            self._generate_diffusion_unet_1d(num_classes=10),
        ]
        
        for model in models:
            self._build_unet_components(model)
            super()._test_train_mode(model)

    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        models = [
            self._generate_diffusion_unet_1d(num_classes=None),
            self._generate_diffusion_unet_1d(num_classes=10),
        ]
        
        for model in models:
            self._build_unet_components(model)
            super()._test_named_parameters_length(model)

    def test_to_device(self):
        """Test that the model can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
        
        models = [
            self._generate_diffusion_unet_1d(num_classes=False),
            self._generate_diffusion_unet_1d(num_classes=10),
        ]

        for model in models:
            self._build_unet_components(model)
            inputs = self._get_valid_input(model)
            
            filtered_inputs = tuple(x for x in inputs if x is not None)
            super()._test_to_device(model, *filtered_inputs)

    
if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main()


