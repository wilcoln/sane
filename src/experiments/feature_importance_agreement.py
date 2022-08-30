import numpy as np
import torch
from torch import nn

from src.settings import settings


def get_batch_jacobian(net, x):
    x.requires_grad_(True)
    y = net(x)
    y = y.max(1)[0]
    y.backward(torch.ones_like(y))
    return x.grad.data


def occlude(inputs, indices):
    """
    Occlude the input at the given indices
    :param inputs: input to occlude
    :param indices: indices to occlude
    :return: occluded input
    """
    inputs = inputs.clone()
    for i, row in enumerate(indices):
        inputs[i, row] = 0
    return inputs.detach()


if __name__ == '__main__':
    # Create model
    num_features = 2 * settings.sent_dim
    model = nn.Linear(num_features, settings.num_classes)

    # Create inputs & targets
    inputs = torch.randn(10, num_features)
    targets = torch.randint(0, settings.num_classes, (10,))

    # Feature importance agreement
    percentiles = np.arange(0, 31, 10)
    influential_occluded_accuracies = []
    random_occluded_accuracies = []
    for percentile in percentiles:
        # Compute gradients
        grad = get_batch_jacobian(model, inputs)
        grad = torch.abs(grad)

        # Compute number of features to occlude
        num_occlude = int(percentile / 100 * num_features)

        # Get most influential features
        topk_attribution_scores, indices = torch.topk(grad, num_occlude, sorted=False)

        # Occlude most influential features
        occluded_inputs = occlude(inputs, indices)

        # Compute accuracy on influentially occluded inputs
        occluded_outputs = model(occluded_inputs)
        occluded_outputs = occluded_outputs.max(1)[1]
        occluded_accuracy = (occluded_outputs == targets).sum().item() / targets.size(0)
        influential_occluded_accuracies.append(occluded_accuracy)

        # Compute accuracy on randomly occluded inputs
        random_indices = torch.randint(0, num_features, (10, num_occlude))
        random_occluded_inputs = occlude(inputs, random_indices)
        random_occluded_outputs = model(random_occluded_inputs)
        random_occluded_outputs = random_occluded_outputs.max(1)[1]
        random_occluded_accuracy = (random_occluded_outputs == targets).sum().item() / targets.size(0)
        random_occluded_accuracies.append(random_occluded_accuracy)

    # Plot results
    import matplotlib.pyplot as plt

    plt.plot(percentiles, influential_occluded_accuracies)
    plt.plot(percentiles, random_occluded_accuracies)
    plt.show()
