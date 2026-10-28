"""
Description: Simple file to create a graph of the transformer loss.
Date: 5/13/25
Author: Brian, August, Thomas
"""

import matplotlib.pyplot as plt

train_losses = [
    0.7212,
    0.0498,
    0.0040,
    0.0008,
    0.0005,
    0.0004,
    0.0003,
    0.0002,
    0.0002,
    0.0001,
]


epochs = list(range(1, len(train_losses) + 1))
plt.plot(epochs, train_losses, color="black", marker="o", label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
