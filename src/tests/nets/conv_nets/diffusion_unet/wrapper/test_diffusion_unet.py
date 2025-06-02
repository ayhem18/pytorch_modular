import torch
import random
import unittest
from typing import Optional, Tuple, List

from tests.custom_base_test import CustomModuleBaseTest
from mypt.nets.conv_nets.diffusion_unet.wrapper.diffusion_unet import DiffusionUNet


class TestDiffusionUNet(CustomModuleBaseTest):
    """Test class for DiffusionUNet with various configuration combinations"""
    
    def setUp(self):
        """Initialize test parameters"""
        # Define common test parameters
        self.input_channels_range = (1, 3)
        self.output_channels_range = (1, 3)
        self.cond_dimension_range = (16, 64)
        self.num_classes_options = [None, 5, 10]
        self.embedding_methods = ['positional', 'gaussian_fourier']
        
        # For 3D condition shapes
        self.condition_3d_channels_range = (1, 3)
        self.condition_3d_height_range = (16, 32)
        self.condition_3d_width_range = (16, 32)
        
        # For UNet blocks
        self.num_resnet_blocks_range = (2, 4)
        self.channels_multiplier_range = (1, 2)
    
    def _get_random_condition_3d_shape(self) -> Tuple[int, int, int]:
        """Generate a random 3D condition shape"""
        channels = random.randint(*self.condition_3d_channels_range)
        height = random.randint(*self.condition_3d_height_range)
        width = random.randint(*self.condition_3d_width_range)
        return (channels, height, width)
    
    def _get_valid_input(self, 
                        model: DiffusionUNet,
                        batch_size: int = 2, 
                        height: int = 64, 
                        width: int = 64) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Generate input tensors appropriate for the model configuration"""

        # Time step tensor (always required)
        time_step = torch.randint(0, 1000, (batch_size,))
        
        # Class tensor (if needed)
        class_tensor = None
        if hasattr(model, 'class_embedding') and model.class_embedding is not None:
            num_classes = model.class_embedding.num_embeddings
            class_tensor = torch.randint(0, num_classes, (batch_size,))
        
        # 3D condition tensor (if needed)
        condition_3d = None
        if hasattr(model, 'embedding_3d_projection') and model.embedding_3d_projection is not None:
            # Create a feature map with appropriate shape
            condition_shape = model.conditions_processor.condition_3d_shape
            condition_3d = torch.randn(batch_size, condition_shape[0], condition_shape[1], condition_shape[2])
            x_shape = (batch_size, model.input_channels, condition_shape[1], condition_shape[2])  
            x = torch.randn(*x_shape)
            
            if model.conditions_processor.condition_3d_label_map:
                condition_3d = torch.clip(condition_3d, min=0, max=num_classes - 1).to(torch.int64)
        else:
            x = torch.randn(batch_size, model.input_channels, height, width)


        return x, time_step, class_tensor, condition_3d
    
    def _build_unet_components(self, model: DiffusionUNet):
        """Build all components of the UNet model"""
        # Random parameters for building
        num_down_layers = random.randint(*self.num_resnet_blocks_range)
        channels_multiplier = 2
        num_res_blocks = random.randint(2, 4)
        
        # Define parameters for down block
        down_params = {
            'num_down_layers': num_down_layers,
            'num_res_blocks': num_res_blocks,
            'out_channels': [32 * channels_multiplier * (2**i) for i in range(num_down_layers)],
            'downsample_types': 'conv',
            'inner_dim': 256,
            'dropout_rate': 0.0,
            'film_activation': 'relu',
            'force_residual': True
        }
        
        # Define parameters for middle block
        mid_params = {
            'num_res_blocks': num_res_blocks,
            'inner_dim': 256,
            'dropout_rate': 0.0,
            'film_activation': 'relu',
            'force_residual': True
        }
        
        # Define parameters for up block (must match down block)
        up_params = {
            'num_res_blocks': num_res_blocks,
            'upsample_types': 'transpose_conv',
            'inner_dim': 256,
            'dropout_rate': 0.0,
            'film_activation': 'relu',
            'force_residual': True
        }
        
        # Build components
        model.build_down_block(**down_params)
        model.build_middle_block(**mid_params)
        model.build_up_block(**up_params)
        
        return model
    
    def _generate_diffusion_unet(self,
                               input_channels=None,
                               output_channels=None,
                               cond_dimension=None,
                               embedding_encoding_method=None,
                               num_classes=None,
                               condition_3d_shape=None,
                               condition_3d_label_map=True) -> DiffusionUNet:
        """Generate a DiffusionUNet with specified or random parameters"""
        # Set random parameters if not provided
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
        
        if num_classes == False: # means not passing class embedding to the DiffusionUNet class
            num_classes = None

        # Create the model
        model = DiffusionUNet(
            input_channels=input_channels,
            output_channels=output_channels,
            cond_dimension=cond_dimension,
            embedding_encoding_method=embedding_encoding_method,
            num_classes=num_classes,
            condition_3d_shape=condition_3d_shape,
            condition_3d_label_map=condition_3d_label_map
        )
        
        return model
    
    ########################## Configuration Tests ##########################
    
    # @unittest.skip("passed")
    def test_initialization_without_3d_without_class(self):
        """Test initialization without 3D condition and without class conditioning"""
        for _ in range(10):
            for encoding in self.embedding_methods:
                model = self._generate_diffusion_unet(
                    embedding_encoding_method=encoding,
                    num_classes=False,
                    condition_3d_shape=None,
                    condition_3d_label_map=False
                )
                
                # Check core components
                self.assertIsNotNone(model.embedding_encoding)
                self.assertIsNotNone(model.embedding_2d_projection)
                self.assertIsNone(model.class_embedding)
                self.assertIsNone(model.embedding_3d_projection)
                
                # Check UNet type
                from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet1d import UNet1DCond
                self.assertIsInstance(model.unet, UNet1DCond)
    

    # @unittest.skip("passed")
    def test_initialization_without_3d_with_class(self):
        """Test initialization without 3D condition but with class conditioning"""
        for _ in range(10):
            for encoding in self.embedding_methods:
                num_classes = random.randint(5, 20)
                model = self._generate_diffusion_unet(
                    embedding_encoding_method=encoding,
                    num_classes=num_classes,
                    condition_3d_shape=None
                )
                
                # Check core components
                self.assertIsNotNone(model.embedding_encoding)
                self.assertIsNotNone(model.embedding_2d_projection)
                self.assertIsNotNone(model.class_embedding)
                self.assertEqual(model.class_embedding.num_embeddings, num_classes)
                self.assertIsNone(model.embedding_3d_projection)
                
                # Check UNet type
                from mypt.nets.conv_nets.diffusion_unet.unet.one_dim.unet1d import UNet1DCond
                self.assertIsInstance(model.unet, UNet1DCond)

    # @unittest.skip("passed")
    def test_initialization_with_3d_without_class(self):
        """Test initialization with 3D condition but without class conditioning"""
        for _ in range(10):
            for encoding in self.embedding_methods:
                condition_3d_shape = self._get_random_condition_3d_shape()
                model = self._generate_diffusion_unet(
                    embedding_encoding_method=encoding,
                    num_classes=False,
                    condition_3d_shape=condition_3d_shape,
                    condition_3d_label_map=False  # Since we don't have class conditioning
                )
                
                # Check core components
                self.assertIsNotNone(model.embedding_encoding)
                self.assertIsNotNone(model.embedding_2d_projection)
                self.assertIsNone(model.class_embedding)
                self.assertIsNotNone(model.embedding_3d_projection)
                
                # Check UNet type
                from mypt.nets.conv_nets.diffusion_unet.unet.three_dim.unet3d import UNet3DCond
                self.assertIsInstance(model.unet, UNet3DCond)
    
    # @unittest.skip("passed")
    def test_initialization_with_3d_with_class(self):
        """Test initialization with 3D condition and with class conditioning"""
        for _ in range(50):
            for encoding in self.embedding_methods:
                condition_3d_shape = self._get_random_condition_3d_shape()
                num_classes = random.randint(5, 20)

                condition_3d_label_map = random.choice([True, False])

                if condition_3d_label_map:
                    condition_3d_shape = list(condition_3d_shape)
                    condition_3d_shape[0] = 1
                    condition_3d_shape = tuple(condition_3d_shape)

                model = self._generate_diffusion_unet(
                    embedding_encoding_method=encoding,
                    num_classes=num_classes,
                    condition_3d_shape=condition_3d_shape,
                    condition_3d_label_map=condition_3d_label_map
                )
                
                # Check core components
                self.assertIsNotNone(model.embedding_encoding)
                self.assertIsNotNone(model.embedding_2d_projection)
                self.assertIsNotNone(model.class_embedding)
                self.assertEqual(model.class_embedding.num_embeddings, num_classes)
                self.assertIsNotNone(model.embedding_3d_projection)
                
                # Check UNet type
                from mypt.nets.conv_nets.diffusion_unet.unet.three_dim.unet3d import UNet3DCond
                self.assertIsInstance(model.unet, UNet3DCond)
    

    ########################## Conditions Processor Tests ##########################

    # @unittest.skip("passed")
    def test_conditions_processor_without_3d_without_class(self):
        """Test conditions_processor with no 3D condition and no class conditioning"""
        for _ in range(50):
            model = self._generate_diffusion_unet(
                condition_3d_shape=None,
                num_classes=False,
            )
            
            # Get test input
            x, timesteps, _, _ = self._get_valid_input(model)
            batch_size = x.shape[0]
            
            processed_cond = model.conditions_processor(timesteps)

            # Check output shape for 1D case (batch_size, cond_dim)
            self.assertEqual(len(processed_cond.shape), 2)
            self.assertEqual(processed_cond.shape[0], batch_size)
            self.assertEqual(processed_cond.shape[1], model.cond_dimension)

    # @unittest.skip("skip for now")
    def test_conditions_processor_without_3d_with_class(self):
        """Test conditions_processor with no 3D condition but with class conditioning"""
        for _ in range(50):
            num_classes = random.randint(5, 20)
            for encoding in self.embedding_methods:
                model = self._generate_diffusion_unet(
                    embedding_encoding_method=encoding,
                    num_classes=num_classes,
                    condition_3d_shape=None
                )
                
                # Get test input
                x, timesteps, class_labels, _ = self._get_valid_input(model)
                batch_size = x.shape[0]
                
                # Process conditions
                processed_cond = model.conditions_processor(timesteps, class_labels)
                
                # Check output shape for 1D case (batch_size, cond_dim)
                self.assertEqual(len(processed_cond.shape), 2)
                self.assertEqual(processed_cond.shape[0], batch_size)
                self.assertEqual(processed_cond.shape[1], model.cond_dimension)

    # @unittest.skip("passed")
    def test_conditions_processor_with_3d_without_class(self):
        """Test conditions_processor with 3D condition but no class conditioning"""
        for _ in range(50):
            condition_3d_shape = self._get_random_condition_3d_shape()
            
            # condition_3d_label_map = random.choice([True, False])

            # if condition_3d_label_map:
            #     condition_3d_shape = list(condition_3d_shape)
            #     condition_3d_shape[0] = 1
            #     condition_3d_shape = tuple(condition_3d_shape)

            model = self._generate_diffusion_unet(
                condition_3d_shape=condition_3d_shape,
                num_classes=False,
                condition_3d_label_map=False
            )
            
            # Get test input
            x, timesteps, _, condition_3d = self._get_valid_input(model)
            batch_size, _, _, _ = x.shape
            
            # Process conditions
            processed_cond = model.conditions_processor(timesteps, condition_3d)
            
            # Check output shape for 3D case (batch_size, cond_dim, height, width)
            self.assertEqual(len(processed_cond.shape), 4)
            self.assertEqual(processed_cond.shape[0], batch_size)
            self.assertEqual(processed_cond.shape[1], model.cond_dimension)
            self.assertEqual(processed_cond.shape[2], condition_3d_shape[1])
            self.assertEqual(processed_cond.shape[3], condition_3d_shape[2])

    # @unittest.skip("passed")
    def test_conditions_processor_with_3d_with_class(self):
        """Test conditions_processor with both 3D condition and class conditioning"""
        for _ in range(50):
            for encoding in self.embedding_methods:
                #  make sure the input is of the form: 2 * unet_down_blocks * k, where k is a random integer
                condition_3d_shape = (16, 128 * random.randint(2, 5), 256 * random.randint(2, 5)) 
                num_classes = random.randint(5, 20)
                
                condition_3d_label_map = random.choice([True, False])
                
                if condition_3d_label_map:
                    condition_3d_shape = list(condition_3d_shape)
                    condition_3d_shape[0] = 1
                    condition_3d_shape = tuple(condition_3d_shape)
                
                model = self._generate_diffusion_unet(
                    embedding_encoding_method=encoding,
                    num_classes=num_classes,
                    condition_3d_shape=condition_3d_shape,
                    condition_3d_label_map=condition_3d_label_map
                )
                
                # Get test input
                x, timesteps, class_labels, condition_3d = self._get_valid_input(model)
                batch_size, _, _, _ = x.shape
                
                # convert the condition_3d into a label map if condition_3d_label_map is True 
                condition_3d = torch.clip(condition_3d, min=0, max=num_classes - 1).to(torch.int64) if condition_3d_label_map else condition_3d

                # Process conditions
                processed_cond = model.conditions_processor(
                    timesteps, 
                    class_labels.to(torch.int64), # torch.nn.Embedding expects the class labels to be of type torch.int64 / torch.long
                    condition_3d
                )
                
                # Check output shape for 3D case (batch_size, cond_dim, height, width)
                self.assertEqual(len(processed_cond.shape), 4)
                self.assertEqual(processed_cond.shape[0], batch_size)
                self.assertEqual(processed_cond.shape[1], model.cond_dimension)
                self.assertEqual(processed_cond.shape[2], condition_3d_shape[1])
                self.assertEqual(processed_cond.shape[3], condition_3d_shape[2])



    ########################## Building Tests ##########################
    
    # @unittest.skip("passed")
    def test_build_components_without_3d(self):
        """Test building UNet components for model without 3D conditioning"""
        for _ in range(10):
            model = self._generate_diffusion_unet(condition_3d_shape=None)
            self._build_unet_components(model)
            
            # Check that components were built
            self.assertIsNotNone(model.unet._down_block)
            self.assertIsNotNone(model.unet._middle_block)
            self.assertIsNotNone(model.unet._up_block)
    
    # @unittest.skip("passed")
    def test_build_components_with_3d(self):
        """Test building UNet components for model with 3D conditioning"""
        for _ in range(10):
            condition_3d_shape = self._get_random_condition_3d_shape()

            condition_3d_label_map = random.choice([True, False])

            if condition_3d_label_map:
                condition_3d_shape = list(condition_3d_shape)
                condition_3d_shape[0] = 1
                condition_3d_shape = tuple(condition_3d_shape)

            num_classes = random.randint(5, 20)
            model = self._generate_diffusion_unet(
                condition_3d_shape=condition_3d_shape,
                num_classes=num_classes,
                condition_3d_label_map=condition_3d_label_map
            )
            self._build_unet_components(model)
            
            # Check that components were built
            self.assertIsNotNone(model.unet._down_block)
            self.assertIsNotNone(model.unet._middle_block)
            self.assertIsNotNone(model.unet._up_block)
    
    ########################## Forward Pass Tests ##########################
    
    # @unittest.skip("passed")
    def test_forward_without_3d_without_class(self):
        """Test forward pass without 3D condition and without class conditioning"""
        for _ in range(10):
            model = self._generate_diffusion_unet(
                num_classes=False,
                condition_3d_shape=None
            )
            self._build_unet_components(model)
            
            # Generate inputs
            x, time_step, _, _ = self._get_valid_input(model)
            
            # Forward pass
            output = model(x, time_step)
            
            # Check output shape
            self.assertEqual(output.shape[0], x.shape[0])  # Batch size
            self.assertEqual(output.shape[1], model.output_channels)  # Output channels
            self.assertEqual(output.shape[2:], x.shape[2:])  # Spatial dimensions
    
    # @unittest.skip("passed")
    def test_forward_without_3d_with_class(self):
        """Test forward pass without 3D condition but with class conditioning"""
        for _ in range(10):
            num_classes = random.randint(5, 20)
            model = self._generate_diffusion_unet(
                num_classes=num_classes,
                condition_3d_shape=None
            )
            self._build_unet_components(model)
            
            # Generate inputs
            x, time_step, class_tensor, _ = self._get_valid_input(model)
            
            # Forward pass
            output = model(x, time_step, class_tensor)
            
            # Check output shape
            self.assertEqual(output.shape[0], x.shape[0])  # Batch size
            self.assertEqual(output.shape[1], model.output_channels)  # Output channels
            self.assertEqual(output.shape[2:], x.shape[2:])  # Spatial dimensions
    

    # @unittest.skip("passed")
    def test_forward_with_3d_without_class(self):
        """Test forward pass with 3D condition but without class conditioning"""
        for _ in range(10):

            #  make sure the input is of the form: 2 * unet_down_blocks * k, where k is a random integer
            condition_3d_shape = (16, 64 * random.randint(2, 5), 64 * random.randint(2, 5)) 

            model = self._generate_diffusion_unet(
                num_classes=False,
                condition_3d_shape=condition_3d_shape,
                condition_3d_label_map=False
            )
            self._build_unet_components(model)
            
            # Generate inputs
            x, time_step, _, condition_3d = self._get_valid_input(model)
            
            # Forward pass
            output = model(x, time_step, condition_3d)
            
            # Check output shape
            self.assertEqual(output.shape[0], x.shape[0])  # Batch size
            self.assertEqual(output.shape[1], model.output_channels)  # Output channels
            self.assertEqual(output.shape[2:], x.shape[2:])  # Spatial dimensions
    
    # @unittest.skip("passed")
    def test_forward_with_3d_with_class(self):
        """Test forward pass with 3D condition and with class conditioning"""
        for _ in range(10):
            condition_3d_shape = (1, 64 * random.randint(2, 5), 64 * random.randint(2, 5)) 
            num_classes = random.randint(5, 20)
            model = self._generate_diffusion_unet(
                num_classes=num_classes,
                condition_3d_shape=condition_3d_shape,
                condition_3d_label_map=True
            )
            self._build_unet_components(model)
            
            # Generate inputs
            x, time_step, class_tensor, condition_3d = self._get_valid_input(model)
            
            condition_3d = torch.clip(condition_3d, min=0, max=num_classes - 1).to(torch.int64)
            # Forward pass
            output = model(x, time_step, class_tensor, condition_3d)
            
            # Check output shape
            self.assertEqual(output.shape[0], x.shape[0])  # Batch size
            self.assertEqual(output.shape[1], model.output_channels)  # Output channels
            self.assertEqual(output.shape[2:], x.shape[2:])  # Spatial dimensions
    
    ########################## CustomModuleBaseTest Tests ##########################
    # @unittest.skip("passed")
    def test_eval_mode(self):
        """Test that the model can be set to evaluation mode"""
        for _ in range(5):
            # Test all four combinations
            models = [
                self._generate_diffusion_unet(num_classes=None, condition_3d_shape=None),
                self._generate_diffusion_unet(num_classes=10, condition_3d_shape=None),
                self._generate_diffusion_unet(num_classes=None, condition_3d_shape=self._get_random_condition_3d_shape(), condition_3d_label_map=False),
                self._generate_diffusion_unet(num_classes=10, condition_3d_shape=(1, 32, 32), condition_3d_label_map=True)
            ]
            
            for model in models:
                self._build_unet_components(model)
                super()._test_eval_mode(model)
    
    # @unittest.skip("passed")
    def test_train_mode(self):
        """Test that the model can be set to training mode"""
        for _ in range(5):
            # Test all four combinations
            models = [
                self._generate_diffusion_unet(num_classes=None, condition_3d_shape=None),
                self._generate_diffusion_unet(num_classes=10, condition_3d_shape=None),
                self._generate_diffusion_unet(num_classes=None, condition_3d_shape=self._get_random_condition_3d_shape(), condition_3d_label_map=False),
                self._generate_diffusion_unet(num_classes=10, condition_3d_shape=(1, 32, 32), condition_3d_label_map=True)
            ]
            
            for model in models:
                self._build_unet_components(model)
                super()._test_train_mode(model)

    # @unittest.skip("")
    def test_named_parameters_length(self):
        """Test that named_parameters and parameters have the same length"""
        for _ in range(5):
            # Test all four combinations
            models = [
                self._generate_diffusion_unet(num_classes=None, condition_3d_shape=None),
                self._generate_diffusion_unet(num_classes=10, condition_3d_shape=None),
                self._generate_diffusion_unet(num_classes=None, condition_3d_shape=self._get_random_condition_3d_shape(), condition_3d_label_map=False),
                self._generate_diffusion_unet(num_classes=10, condition_3d_shape=(1, 32, 32), condition_3d_label_map=True)
            ]
            
            for model in models:
                self._build_unet_components(model)
                super()._test_named_parameters_length(model)
    
    # @unittest.skip("skipping test_to_device")
    def test_to_device(self):
        """Test that the model can be moved between devices"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device tests")
            
        for _ in range(5):
            c_shape = (1, 64, 64)
            
            models = [
                self._generate_diffusion_unet(num_classes=None, condition_3d_shape=None),
                self._generate_diffusion_unet(num_classes=10, condition_3d_shape=None),
                self._generate_diffusion_unet(num_classes=None, condition_3d_shape=(3,) + c_shape[1:], condition_3d_label_map=False),
                self._generate_diffusion_unet(num_classes=10, condition_3d_shape=c_shape, condition_3d_label_map=True)
            ]

            for i, model in enumerate(models):
                self._build_unet_components(model)
                inputs = self._get_valid_input(model)
                
                # For each configuration, we need to filter out None inputs
                filtered_inputs = tuple(x for x in inputs if x is not None)
                super()._test_to_device(model, *filtered_inputs)



if __name__ == '__main__':
    import mypt.code_utils.pytorch_utils as pu
    pu.seed_everything(42)
    unittest.main()
