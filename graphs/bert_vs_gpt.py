"""
Description: Simple file to create a graph of the performance of CodeBERT vs GPT-2.
Date: 5/13/25
Author: Brian, August, Thomas
"""

import matplotlib.pyplot as plt

bert_acc = 0.9762
t_acc = 0.9492

plt.bar(["CodeBERT", "GPT-2"], [bert_acc, t_acc], color="skyblue")
plt.xlabel("Tokenizer", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0.94, 0.98)
plt.show()
