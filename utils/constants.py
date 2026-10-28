"""
Description: A file to keep track of constants used throughout.
Date: 5/13/25
Author: Brian, August, Thomas
"""

import torch
from transformers import AutoTokenizer

# The supported device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # gpt2 tokenizer - can also switch to use codeBERT tokenizer below
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token

# CodeBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Constants used for the models
seq_len = 1024
batch_size = 4
vocab_size = tokenizer.vocab_size
num_epochs = 10
lr = 1e-4
embed_dimension = 512
hidden_dimension = 1024
