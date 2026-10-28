"""
Description: This file is used to evaluate the transformer model. It simply loads the model using the saved model parameters and then iterates through the validation data and determines the model accuracy.
Date: 5/13/25
Author: Brian, August, Thomas
"""

from utils.constants import vocab_size, device
from utils.data_loader import val_loader
from utils.tokenizer import tokenizer
from transformer.transformer import *
import torch.nn.functional as F
import torch

# Load the model from the saved params and set to eval mode
model = Transformer(vocab_size).to(device)
model.load_state_dict(torch.load("transformer_model.pt"))
model.eval()

correct_tokens = 0
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

        # Chooses most likely output vocab index, makes sure it was not a padding token
        predictions = output.argmax(dim=-1)
        padding = x != tokenizer.pad_token_id
        correct = (predictions == x) & padding

        correct_tokens += correct.sum().item()
        total_tokens += padding.sum().item()

accuracy = correct_tokens / total_tokens
print(f"Validation Accuracy: {accuracy:.4f}")
