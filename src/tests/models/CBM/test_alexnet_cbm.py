import torch
import unittest

from torch import nn

from mypt.backbones.alexnetFE import AlexNetFE
from mypt.models.CBM.alexnet_cbm import AlexnetCBM
from tests.custom_base_test import CustomModuleBaseTest
from mypt.building_blocks.linear_blocks.fc_blocks import ExponentialFCBlock


class TestAlexnetCBM(CustomModuleBaseTest):
    """
    Test class for AlexnetCBM implementation.
    Tests structure, component initialization, and field settings.
    """

    def _get_valid_input(self) -> torch.Tensor:
        """
        Generate a random input tensor with the correct shape for testing.
        """
        return torch.rand(2, 3, 224, 224)  # batch_size, channels, height, width

    def _create_default_cbm_model(self) -> AlexnetCBM:
        """
        Create a default AlexnetCBM model for testing.
        """
        return AlexnetCBM(
            input_shape=(3, 224, 224),
            num_concepts=10,
            num_classes=5,
            alexnet_fe_blocks='conv_block',
            alexnet_fe_frozen_blocks=['conv1', 'conv2'],
        )

    def test_init_structure(self):
        """
        Test that the AlexnetCBM model has the correct structure after initialization.
        """
        model = self._create_default_cbm_model()
        
        # Test class type
        self.assertIsInstance(model, AlexnetCBM)
        
        # Test components structure
        self.assertIsInstance(model._feature_extractor, AlexNetFE)
        self.assertIsInstance(model._concept_projection, ExponentialFCBlock)
        self.assertIsInstance(model._classification_head, ExponentialFCBlock)
        self.assertIsInstance(model._flatten_layer, nn.Flatten)

        # Test that the model field has the correct structure
        self.assertIsInstance(model._model, nn.Sequential)
        self.assertEqual(len(model._model), 4)  # feature_extractor, flatten_layer, concept_projection, classification_head
        self.assertIs(model._model[0], model._feature_extractor)
        self.assertIs(model._model[1], model._flatten_layer)
        self.assertIs(model._model[2], model._concept_projection)
        self.assertIs(model._model[3], model._classification_head)

    def test_feature_extractor_initialization(self):
        """
        Test that the feature extractor is initialized correctly.
        """
        model = self._create_default_cbm_model()
        feature_extractor = model._feature_extractor
        
        # Test feature extractor type
        self.assertIsInstance(feature_extractor, AlexNetFE)
        
        # Test different initialization configurations
        # Test with different block specifications
        block_specs = [
            'conv_block_avgpool',
            'conv3',
            3,
        ]
        
        for blocks in block_specs:
            test_model = AlexnetCBM(
                input_shape=(3, 224, 224),
                num_concepts=10,
                num_classes=5,
                alexnet_fe_blocks=blocks,
                alexnet_fe_frozen_blocks=False
            )
            self.assertIsInstance(test_model._feature_extractor, AlexNetFE)


    def test_concept_projection_initialization(self):
        """
        Test that the concept projection layer is initialized correctly.
        """
        model = self._create_default_cbm_model()
        concept_projection = model._concept_projection
        
        # Test concept projection type
        self.assertIsInstance(concept_projection, ExponentialFCBlock)
        
        # Test concept projection configuration
        self.assertEqual(concept_projection.output, model._num_concepts)
        self.assertEqual(concept_projection.num_layers, 2)  # default value

    def test_classification_head_initialization(self):
        """
        Test that the classification head is initialized correctly.
        """
        model = self._create_default_cbm_model()
        classification_head = model._classification_head
        
        # Test classification head type
        self.assertIsInstance(classification_head, ExponentialFCBlock)
        
        # Test classification head configuration
        self.assertEqual(classification_head.output, model._output_units)
        self.assertEqual(classification_head.in_features, model._num_concepts)
        self.assertEqual(classification_head.num_layers, 2)  # default value

    def test_forward_pass_output_shape(self):
        """
        Test that the forward pass returns outputs with the correct shape.
        """
        model = self._create_default_cbm_model()
        model.eval()  # Set to evaluation mode
        
        batch_size = 2
        input_tensor = torch.rand(batch_size, 3, 224, 224)
        concept_logits, class_logits = model.forward(input_tensor)
        
        # Test output shapes
        self.assertEqual(concept_logits.shape, (batch_size, model._num_concepts))
        self.assertEqual(class_logits.shape, (batch_size, model._output_units))

    def test_output_units_calculation(self):
        """
        Test that the output_units is calculated correctly based on num_classes.
        """
        # For binary classification (num_classes=2), output_units should be 1
        binary_model = AlexnetCBM(
            input_shape=(3, 224, 224),
            num_concepts=10,
            num_classes=2,
            alexnet_fe_blocks='conv_block',
            alexnet_fe_frozen_blocks=False,
        )
        self.assertEqual(binary_model._output_units, 1)
        
        # For multi-class classification, output_units should equal num_classes
        multi_model = AlexnetCBM(
            input_shape=(3, 224, 224),
            num_concepts=10,
            num_classes=5,
            alexnet_fe_blocks='conv_block',
            alexnet_fe_frozen_blocks=False,
        )
        self.assertEqual(multi_model._output_units, 5)

    def test_custom_modules(self):
        """
        Test that custom modules can be passed and used correctly.
        """
        custom_concept_projection = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 15)
        )
        
        custom_classification_head = nn.Sequential(
            nn.Linear(15, 3)
        )
        
        model = AlexnetCBM(
            input_shape=(3, 224, 224),
            num_concepts=15,
            num_classes=3,
            alexnet_fe_blocks='conv_block',
            alexnet_fe_frozen_blocks=False,
            concept_projection=custom_concept_projection,
            classification_head=custom_classification_head
        )
        
        # Test that custom modules are used
        self.assertIs(model._concept_projection, custom_concept_projection)
        self.assertIs(model._classification_head, custom_classification_head)

    def test_frozen_blocks_configuration(self):
        """
        Test different frozen block configurations.
        """
        frozen_configs = [
            True,  # Freeze all blocks
            False,  # Don't freeze any blocks
            'conv2',  # Freeze up to conv2
            2,  # Freeze up to block 2
            ['conv1', 'conv3'],  # Freeze specific blocks by name
            [0, 2]  # Freeze specific blocks by index
        ]
        
        for frozen_config in frozen_configs:
            model = AlexnetCBM(
                input_shape=(3, 224, 224),
                num_concepts=10,
                num_classes=5,
                alexnet_fe_blocks='conv_block_avgpool',
                alexnet_fe_frozen_blocks=frozen_config
            )
            self.assertIsInstance(model._feature_extractor, AlexNetFE)
            # The test passes if initialization succeeds without errors

    # CustomModuleBaseTest methods
    def test_eval_mode(self):
        """Test that calling eval() sets training=False for all parameters and submodules"""
        block = self._create_default_cbm_model()
        self._test_eval_mode(block)
    
    def test_train_mode(self):
        """Test that calling train() sets training=True for all parameters and submodules"""
        block = self._create_default_cbm_model()
        self._test_train_mode(block)
    
    def test_consistent_output_in_eval_mode(self):
        """Test that all modules in eval mode produce consistent output for the same input"""
        block = self._create_default_cbm_model()
        input_tensor = self._get_valid_input()
        self._test_consistent_output_in_eval_mode(block, input_tensor)
    
    def test_batch_size_one_in_eval_mode(self):
        """Test that modules in eval mode should not raise errors for batch size 1"""
        block = self._create_default_cbm_model()
        input_tensor = torch.rand(1, 3, 224, 224)
        self._test_batch_size_one_in_eval_mode(block, input_tensor)
    
    def test_batch_size_one_in_train_mode(self):
        """Test that modules in train mode should not raise errors for batch size 1"""
        block = self._create_default_cbm_model()
        input_tensor = torch.rand(1, 3, 224, 224)
        self._test_batch_size_one_in_train_mode(block, input_tensor)

    def test_consistent_output_without_dropout_bn(self):
        """Test that modules should produce consistent output without dropout and batch norm"""
        block = self._create_default_cbm_model()
        input_tensor = self._get_valid_input()
        self._test_consistent_output_without_dropout_bn(block, input_tensor)

    def test_named_parameters_length(self):
        """Test that named_parameters() and parameters() have the same length"""
        block = self._create_default_cbm_model()
        self._test_named_parameters_length(block)

    def test_to_device(self):
        """Test that module can move between devices properly"""
        block = self._create_default_cbm_model()
        input_tensor = self._get_valid_input()
        self._test_to_device(block, input_tensor)


if __name__ == '__main__':
    unittest.main()