import torch

class TransformerClassifier(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, value_dim: int, key_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.key_dim = key_dim
        