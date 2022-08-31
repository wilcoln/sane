import numpy as np
import torch
from torch import nn

from src.settings import settings
from src.utils.embeddings import corrupt

if __name__ == '__main__':
    # Create model
    num_features = 2 * settings.sent_dim
    model = nn.Linear(num_features, settings.num_classes)

    # Create inputs & targets
    inputs = torch.randn(10, num_features)
    targets = torch.randint(0, settings.num_classes, (10,))

    # Robustness equivalence
    sigma2s = np.arange(0, 31, 5)
    accuracies = []
    for sigma2 in sigma2s:
        # Compute gradients
        corrupted_inputs = corrupt(inputs, sigma2=sigma2).detach()

        # Compute accuracy on randomly occluded inputs
        outputs = model(corrupted_inputs)
        outputs = outputs.max(1)[1]
        accuracy = (outputs == targets).sum().item() / targets.size(0)
        accuracies.append(accuracy)

    # Plot results
    import matplotlib.pyplot as plt

    plt.plot(sigma2s, accuracies)
    if settings.show_plots:
        plt.show()
