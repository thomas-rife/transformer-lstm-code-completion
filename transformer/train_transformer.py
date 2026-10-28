"""
Description: This file is used for training the model. It iterates through all the training data in batches and trains the model. Then it saves the model parameters so that it can be be loaded.
Date: 5/13/25
Author: Brian, August, Thomas
"""

from utils.data_loader import train_loader
from utils.tokenizer import tokenizer
from transformer.transformer import *
import torch.optim as optim
import torch.nn.functional as F
import torch
from utils.constants import vocab_size, lr, device, num_epochs

# Build the model
model = Transformer(vocab_size).to(device)

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

def compute_loss(output, target):
    """
    Computes the cross-entropy loss between predicted output and target tokens.

    Parameters
    --------------------
        output -- tensor of shape (batch_size, seq_len, vocab_size)
        target -- tensor of shape (batch_size, seq_len)

    Returns
    --------------------
        loss -- cross-entropy loss, ignoring padding tokens
    """
    loss = F.cross_entropy(
        output.view(-1, vocab_size),  # flatten dimensions (batch * seq_len, vocab_size)
        target.view(-1),  # batch * seq_len
        ignore_index=tokenizer.pad_token_id,
    )
    return loss


# Training loop
for epoch in range(num_epochs):
    model.train()

    for x, attention_mask in train_loader:
        x = x.to(device)
        attention_mask = attention_mask.to(device)
        optimizer.zero_grad()
        output = model(x, attention_mask)
        loss = compute_loss(output, x)
        loss.backward()
        optimizer.step()
        print(loss.item())

# Save the model parameters
torch.save(model.state_dict(), "transformer_model.pt")
