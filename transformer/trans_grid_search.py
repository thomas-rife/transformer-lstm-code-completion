"""
Description: This file is used for finding the best hyperparameters for the transformer model.
Date: 5/13/25
Author: Brian, August, Thomas
"""

from utils.data_loader import train_loader, val_loader
from utils.tokenizer import tokenizer
from transformer.transformer import *
from transformer.train_transformer import compute_loss
from utils.constants import vocab_size, device

import torch.optim as optim
import torch.nn.functional as F
import torch
import itertools
import math

num_epochs = 2

# parameters for model: num_layers, learning rate, dim_feedforward
vals_num_layers = (2, 4, 8)
vals_learning_rate = (1e-3, 1e-4, 1e-5)
vals_dim_feedforward = (512, 1024, 2048)

values = itertools.product(vals_num_layers, vals_learning_rate, vals_dim_feedforward)

for num_layers, lr, dim_ff in values:
    print("Current model params", num_layers, lr, dim_ff)

    # Build the model
    model = Transformer(vocab_size, num_layers=num_layers, dim_feedforward=dim_ff).to(
        device
    )

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

    model.eval()

    correct_tokens = 0
    total_loss = 0
    total_tokens = 0

    # Evaluation loop, goes through each batch in val_loader and counts how many predictions the model gets right.
    with torch.no_grad():
        for x, attention_mask in val_loader:
            x = x.to(device)
            attention_mask = attention_mask.to(device)
            output = model(x, attention_mask)

            # Calculates the loss value
            loss = F.cross_entropy(
                output.view(
                    -1, tokenizer.vocab_size
                ),  # flatten dimensions (batch * seq_len, vocab_size)
                x.view(-1),  # batch * seq_len
                ignore_index=tokenizer.pad_token_id,
            )

            # add to total loss
            total_loss += loss.item()

            # Chooses most likely output vocab index, makes sure it was not a padding token
            predictions = output.argmax(dim=-1)
            padding = x != tokenizer.pad_token_id
            correct = (predictions == x) & padding

            correct_tokens += correct.sum().item()
            total_tokens += padding.sum().item()

    accuracy = correct_tokens / total_tokens
    print(
        f"Validation Accuracy: {accuracy:.4f} WITH layers={num_layers}, lr={lr}, dim_ff={dim_ff}"
    )

    # Calculate perplexity
    perplexity = math.exp(total_loss / total_tokens)
    print(f"Test Perplexity: {perplexity:.4f}")

    del model
    torch.cuda.empty_cache()
