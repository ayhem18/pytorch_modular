import torch 
from torch import nn

class CosineSim(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x:torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # convert the input to float if needed 
        x, y = x.to(torch.float32), y.to(torch.float32)

        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"This function expects the input to be 2 dimensional. Found: x: {x.shape}, y: {y.shape}")

        if x.shape[1] != y.shape[1]:
            raise ValueError(f"x and y must be of the same dimensions !!. Found: x: {x.shape}, y: {y.shape}")

        x_samples, y_samples = x.shape[0], y.shape[0]

        norm_x = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        norm_y = torch.linalg.vector_norm(y, dim=1, keepdim=True)
        
        # normalize both vectors
        x_norm = x / norm_x
        y_norm = y / norm_y 

        # cosine similarity is basically dot product of normalized vectors...
        res = x_norm @ y_norm.T
        # make sure the loss is of the
        assert res.shape == (x_samples, y_samples), f"Make sure the final result matches the expected shape. Expected: {(x_samples, y_samples)}. Found: {res.shape}"
        return res
