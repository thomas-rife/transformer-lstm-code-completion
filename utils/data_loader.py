"""
Description: File to create the DataLoaders for both training and testing.
Date: 5/13/25
Author: Brian, August, Thomas
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils.constants import batch_size


class GetDataset(Dataset):

    def __init__(self, file_paths):
        """
        A dataset for loading pre-tokenized data from a list of file paths.

        Attributes
        --------------------
            file_paths -- list of file paths containing pre-tokenized code as tensors.
        """
        self.file_paths = file_paths

    def __len__(self):
        """
        Returns the total number of files in the dataset.

        Returns
        --------------------
            int -- the number of files in the dataset
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads the tensor from the file at the given index.

        Parameters
        --------------------
            idx   -- index of the file to load

        Returns
        --------------------
            tensor -- the tensor from the specified file
        """
        file_path = self.file_paths[idx]
        tensor = torch.load(file_path)
        return tensor


def load_data_from_dir(data_dir, batch_size):
    file_paths = list(Path(data_dir).glob("*.pt"))
    dataset = GetDataset(file_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


train_loader = load_data_from_dir("data/python/train", batch_size=batch_size)
val_loader = load_data_from_dir("data/python/val", batch_size=batch_size)
test_loader = load_data_from_dir("data/python/test", batch_size=batch_size)
