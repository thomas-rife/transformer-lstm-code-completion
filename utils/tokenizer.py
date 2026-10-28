"""
Description: File for tokenizing a code string.
Date: 5/13/25
Author: Brian, August, Thomas
"""

from utils.constants import tokenizer, seq_len


def tokenize_code(code):
    """
    Takes in Python source code as a string and returns tokenized tensors for model input.

    Parameters
    --------------------
        code    -- string that represents Python program code

    Returns
    --------------------
        input_ids       -- tensor of shape (seq_len) containing tokenized code, where tokens are padded or truncated to match the fixed sequence length (seq_len). Padding tokens are added to the right if needed.

        attention_mask  -- tensor of shape (seq_len,) containing binary values indicating which tokens are actual content. A value of 1 means the token is real (not padding), and 0 means it is a padding token.
    """
    tokens = tokenizer.encode_plus(
        code,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].squeeze(0)
    attention_mask = tokens["attention_mask"].squeeze(0)
    return input_ids, attention_mask
