"""
Description: Main class for the transformer model.
Date: 5/13/25
Author: Brian, August, Thomas
"""

import torch
import torch.nn as nn
from utils.constants import seq_len, batch_size


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers=6,
        d_model=512,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=seq_len,
    ):
        """
        A Transformer-based language model for code completion.

        Attributes
        --------------------
            token_embeddings -- embedding layer for token IDs
            position_embeddings -- embedding layer for positional encodings
            encoder -- stack of Transformer encoder layers
            output_layer -- linear layer going from hidden states to vocab size
            max_len -- maximum sequence length
            d_model -- embedding dimension
        """
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, attention_mask):
        """
        Performs a forward pass of the model.

        Parameters
        --------------------
            x              -- tensor of shape (batch_size, seq_len) containing token indices
            attention_mask -- tensor of shape (batch_size, seq_len), with 1 for real tokens and 0 for padding

        Returns
        --------------------
            tensor -- output logits of shape (batch_size, seq_len, vocab_size)
        """
        positions = (
            torch.arange(seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        x = self.token_embeddings(x) + self.position_embeddings(positions)

        # Create causal mask (prevents attending to future tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            x.device
        )

        # Padding mask: True for padding
        key_padding_mask = (attention_mask == 0).to(torch.float)

        encoded = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        return self.output_layer(encoded)
