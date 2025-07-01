"""
This script contains the implementation of a masked language model based on the pytorch decoder layer implementation.
"""

import torch
from torch import nn
from typing import Optional

from mypt.building_blocks.auxiliary.embeddings.scalar.encoding import PositionalEncoding
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin


class MaskedLM(NonSequentialModuleMixin, nn.Module):
    """
    A Transformer-based Masked Language Model.

    This class uses `torch.nn.TransformerEncoderLayer` and `torch.nn.TransformerEncoder`
    for the main transformer blocks.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the MaskedLM.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The number of expected features in the input.
            num_layers (int): The number of sub-encoder-layers in the encoder.
            num_heads (int): The number of heads in the multiheadattention models.
            dropout (float): The dropout value. Defaults to 0.1.
        """
        nn.Module.__init__(self)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=['embedding', 'pos_emb', 'encoder', 'head'])

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Linear(d_model, vocab_size)


    def forward(self, sequence: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Masked Language Model.

        Args:
            sequence (torch.Tensor): Input sequence tensor of shape (batch, seq_len) with token IDs.
            pad_mask (Optional[torch.Tensor]): Padding mask of shape (batch, seq_len),
                                              where `1` indicates a valid token and `0` a pad token.

        Returns:
            torch.Tensor: The output logits from the model of shape (batch, seq_len, vocab_size).
        """
        # convert tokens ids to embeddings
        x = self.embedding(sequence)
        # compute position encoding
        pos_encoding = self.pos_emb(torch.arange(sequence.shape[1], device=sequence.device))
        # mask the position encoding if there is a padding mask
        if pad_mask is not None:
            pos_encoding = pos_encoding.masked_fill(~pad_mask.bool().unsqueeze(-1), 0)
        # add the position encoding to the embeddings
        x = x + pos_encoding

        # create a key padding mask for the encoder
        src_key_padding_mask = None
        if pad_mask is not None:
            # src_key_padding_mask is a boolean mask where True means ignore
            # pad_mask is a boolean mask where False means ignore, should be inverted before passing it to the encoder.
            src_key_padding_mask = (pad_mask == False)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        result = self.head(x)
        
        return result

    def __call__(self, sequence: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(sequence, pad_mask)