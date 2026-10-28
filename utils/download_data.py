"""
Description: File for downloading data into train and validation folders.
Date: 5/13/25
Author: Brian, August, Thomas
"""

from datasets import load_dataset
from pathlib import Path
import hashlib
from utils.tokenizer import tokenize_code
from utils.constants import seq_len
import torch

"""
This file is intended to be run once in order to download data, processed to be tensors. Directions for running can be found in the README.
"""

if __name__ == "__main__":
    dataset = load_dataset(
        "codeparrot/github-code",
        split="train",
        streaming=True,
        languages=["Python"],
        trust_remote_code=True,
    )

    # Defining paths for where the data will be stored
    train_dir = Path("data/python/train")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir = Path("data/python/val")
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir = Path("data/python/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    def sliding_window(snippet):
        """
        Implements a sliding window with 25 percent overlap between sequences.

        Parameters
        --------------------
            snippet -- a string of code

        Returns
        --------------------
            seqs -- sequences of the code, broken into seq len or less with 25 percent overlap
        """
        stride = int(seq_len * 0.75)
        seqs = []
        length = len(snippet)

        if length <= seq_len:
            return [snippet]

        for i in range(0, length, stride):
            seqs.append(snippet[i : i + seq_len])
        return seqs

    # Number of files that one wants to write. Datasets contains like 100GB of files
    MAX_FILES = 100
    written = 0

    for i, file_record in enumerate(dataset):

        if written >= MAX_FILES:
            break
        if file_record.get("language") != "Python":
            continue
        snippet = file_record["code"]
        if not snippet.strip():
            continue

        for seq in sliding_window(snippet):
            tokens = tokenize_code(seq)

            # build an id string from the repo and path and turn it into filename
            id_str = f"{file_record['repo_name']}::{file_record['path']}"
            file_name = hashlib.md5(seq.encode("utf-8")).hexdigest() + ".pt"

            # TODO: Fix so that the data is split in a better way
            if written % 10 == 8:
                dest_file = val_dir / file_name
            elif written % 10 == 9:
                dest_file = test_dir / file_name
            else:
                dest_file = train_dir / file_name

            # don't write duplicates
            if not dest_file.exists():
                torch.save(tokens, dest_file)
                print(f"[{i}] wrote → {dest_file.name}")

        written += 1
