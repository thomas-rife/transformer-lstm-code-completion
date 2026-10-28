"""
Description: This module defines an LSTM-based language model for token 
prediction and provides functionality to train and evaluate it on tokenized
Python code datasets loaded from .pt files. Designed to work with Cuda enabled devices
Date: 4/16/25
Author: Brian, August, Thomas
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

from utils.tokenizer import tokenize_code
from utils.constants import vocab_size, num_epochs, lr, embed_dimension, hidden_dimension
from utils.data_loader import load_data_from_dir, train_loader, val_loader, test_loader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

class LSTM(nn.Module):
    '''
    LSTM-based neural network for sequence modeling.

    Args:
        input_size (int): Size of the input vocabulary.
        emb_dim (int): Dimension of the token embeddings.
        hidden_size (int): Number of features in the hidden state of the LSTM.
        num_layers (int): Number of recurrent layers.
        output_size (int): Size of the output vocabulary (same as input_size).
        batch_first (bool): If True, input and output tensors are provided as (batch, seq, feature).
    '''
    def __init__(self, input_size, emb_dim, hidden_size, num_layers, output_size, batch_first=True):
        super().__init__()
        self.emb = nn.Embedding(input_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, bidirectional=False, dropout = 0.1, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''
        Forward pass through the network.

        Args:
            x (torch.LongTensor): Tensor of input token indices with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Unnormalized scores for each token in the output vocabulary,
                            shape (batch_size, sequence_length, output_size).
        '''
        embeds = self.emb(x)
        output, (hidden, cell) = self.lstm(embeds)
        linear_trans = self.linear(output)
        return linear_trans

def main():
    '''
    Main training and evaluation loop.

    Loads tokenized training and validation data, initializes the model, optimizer, and loss function,
    and runs the training loop for a specified number of epochs. After training, evaluates the model
    on the validation set and reports token-level accuracy. Performs gridsearch on a list of layers
    as well as a list of learning rates.
    '''

    layer_choices = [2, 3, 4, 6]
    lr_choices    = [1e-3, 1e-4, 5e-4]

    best_acc = 0.0
    best_cfg = None

    for nl in layer_choices:
        for g_lr in lr_choices:
            print(f"\n→ Testing nl={nl}, lr={g_lr:.1e}")

            # Instantiate model & optimizer each time
            model = LSTM(
                vocab_size,
                embed_dimension,
                hidden_dimension,
                nl,                  # variable num_layers
                vocab_size
            ).to(device)

            # Initialize optimizer with all learning rates
            optimizer = optim.Adam(model.parameters(), lr=g_lr)
            loss_fn = nn.CrossEntropyLoss(reduction='mean')

            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                total_tokens = 0
                batch_count = 0
                train_losses = []
                loop = tqdm(train_loader,
                            desc=f"Epoch {epoch+1}/{num_epochs}",
                            unit="batch", leave=False)
                for input_ids, _ in loop:
                    input_ids = input_ids.to(device)
                    preds = model(input_ids[:, :-1])
                    loss = loss_fn(preds.transpose(1, 2), input_ids[:, 1:])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    batch_count += 1
                    loop.set_postfix(avg_loss=running_loss/batch_count)
                epoch_avg = running_loss / batch_count
                train_losses.append(epoch_avg)

                # Print the average loss per batch
                print(f"Epoch {epoch+1}/{num_epochs} avg loss (per batch): {epoch_avg:.4f}")

            # Eval Loop
            model.eval()
            y_true, y_pred = [], []
            total_loss = 0.0
            total_tokens = 0

            with torch.no_grad():
                for input_ids, _ in val_loader:
                    input_ids = input_ids.to(device)
                    logits = model(input_ids[:, :-1])
                    preds  = logits.argmax(dim=-1)
                    y_true.append(input_ids[:,1:].cpu())
                    y_pred.append(preds.cpu())
                    targets   = input_ids[:, 1:]

                    loss = loss_fn(logits.transpose(1,2),targets)
                    total_loss   += loss.item()
                    total_tokens += targets.numel()

            y_true_flat = torch.cat(y_true, dim=0).reshape(-1).numpy()
            y_pred_flat = torch.cat(y_pred, dim=0).reshape(-1).numpy()
            acc = accuracy_score(y_true_flat, y_pred_flat)

            # Print final training loss
            print(f"\nFinal training loss after {num_epochs} epochs: {train_losses[-1]:.4f}")

            # Perpelxity Calcuation
            avg = total_loss / total_tokens
            perplexity = math.exp(avg)

            print(f"Validation Perplexity: {perplexity:.4f}")

            print(f"  → val acc = {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_cfg = (nl, g_lr)
                torch.save(model.state_dict(), "lstm_model_best.pt")

    print(f"\nBest config: layers={best_cfg[0]}, lr={best_cfg[1]:.1e} → acc={best_acc:.4f}, average loss: {avg:.4f}")

main()