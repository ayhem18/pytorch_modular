"""
This script contains a few checks to make sure that a given dataset or model are compatible with the BCE loss function. 

- for a model: the result of BCE(Sigmoid(model(x)), y) should be the same BCEwithLogits(model(x), y)
- for a dataset: the labels must be binary.
"""

import torch

import torch.nn.functional as F

from typing import Callable, List, Tuple

def model_sanity_check(model: torch.nn.Module, input_shape: Tuple[int], output_shape: Tuple[int], num_tests=100, device=None):
    """
    Thoroughly checks if a model is compatible with BCEWithLogits loss by comparing:
    BCE(Sigmoid(model(x)), y) vs BCEWithLogits(model(x), y) across multiple random inputs.
    
        model: PyTorch model to check
    Args:
        input_shape: Shape of the input tensor (excluding batch dimension)
        output_shape: Shape of the output tensor (excluding batch dimension)
        num_tests: Number of random tests to perform
        device: Device to run the test on (None for current device)
        
    Returns:
        bool: True if the model passes all sanity checks, False otherwise
        
    Raises:
        ValueError: If the model consistently produces outputs that don't work with BCE loss
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Determine device
    if device is None:
        device = next(model.parameters()).device
    
    
    for i in range(num_tests):
        # Generate random input and target
        sample_input = torch.rand(1, *input_shape, device=device)
        sample_target = torch.randint(0, 1, (1, *output_shape), device=device).float()
        
        # Get model output
        with torch.no_grad():
            model_output = model(sample_input)
            
            # Verify output shape matches expected
            if model_output.shape[1:] != torch.Size(output_shape):
                raise ValueError(f"Model output shape {model_output.shape} doesn't match expected {(1, *output_shape)}")
        
        # Method 1: BCE with sigmoid
        bce_loss = F.binary_cross_entropy(torch.sigmoid(model_output), sample_target)
        
        # Method 2: BCE with logits
        bce_with_logits_loss = F.binary_cross_entropy_with_logits(model_output, sample_target)
        
        # Compare the two losses
        is_close = torch.isclose(bce_loss, bce_with_logits_loss)

        if not is_close:
            raise ValueError("The output of the model should be fed to the BCEWithLogits loss function. It might have an activation function applied at the end... Please check")



def dataset_sanity_check(ds: torch.utils.data.Dataset, label_extractor: Callable):
    """
    Checks if a dataset is compatible with BCE loss by verifying labels are binary (0 or 1).
    
    Args:
        ds: PyTorch dataset to check
        label_extractor: Callable that extracts labels from ds[i]
        
    Returns:
        bool: True if the dataset passes the sanity check, False otherwise
    """

    for idx in range(len(ds)):
        # Extract labels using the provided extractor function
        labels = label_extractor(ds[idx])
        
        if not isinstance(labels, (List, Tuple)):
            labels = [labels]


        for l in labels:
            if not isinstance(l, (int, float, torch.Tensor)):
                raise ValueError("Labels must be a list of ints, floats, or tensors")

            if isinstance(l, torch.Tensor):
                binary_tensor = torch.all(torch.logical_or(l == 0, l == 1))
                
                if not binary_tensor:
                    raise ValueError("Labels must be binary (0 or 1)")

            # at this point the label is a scalar
            if int(l) not in [0, 1]:
                raise ValueError("Labels must be binary (0 or 1)")



