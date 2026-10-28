"""
Description: Simple file to create a graph of the LSTM loss.
Date: 5/13/25
Author: Brian, August, Thomas
"""

import matplotlib.pyplot as plt

train_losses = [
    2.0221,
    1.4442,
    1.1879,
    0.9974,
    0.8475,
    0.7215,
    0.6130,
    0.5214,
    0.4414,
    0.3721,
]

epochs = list(range(1, len(train_losses) + 1))
plt.plot(epochs, train_losses, color="black", marker="o", label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
