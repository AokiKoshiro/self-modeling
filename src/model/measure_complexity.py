import copy

import numpy as np
import torch

from src.common.tools import get_device, load_config
from src.data.load import get_data_loaders
from src.model.evaluate import evaluate


def measure_weight_distribution(model):
    final_layer = model.output
    weights = final_layer.weight.data.cpu().numpy()
    std_dev = np.std(weights)
    return std_dev


def prune_model(model, threshold):
    pruned_model = copy.deepcopy(model)
    final_layer = pruned_model.output
    weights = final_layer.weight.data
    mask = torch.abs(weights) > threshold
    weights *= mask
    return pruned_model


def simpson_rule(y):
    n = len(y)
    h = 1 / (n - 1)
    sparsity = 0
    for i in range(n):
        if i == 0 or i == n - 1:
            sparsity += y[i]
        elif i % 2 == 0:
            sparsity += 2 * y[i]
        else:
            sparsity += 4 * y[i]
    sparsity *= h / 3
    return sparsity


def measure_sparsity(model, pruning_interval=0.1):
    config = load_config()
    device = get_device(config["device"])
    _, test_loader = get_data_loaders()

    final_layer = model.output
    weights = final_layer.weight.data.cpu().numpy().flatten()
    sorted_weights = np.sort(np.abs(weights))
    total_weights = len(sorted_weights)

    sparsity = 0
    accuracies = []
    remaining_rate = []

    for i in np.arange(1, 0, -pruning_interval):
        threshold = sorted_weights[int((1 - i) * total_weights)]
        pruned_model = prune_model(model, threshold)
        accuracy = evaluate(pruned_model, device, test_loader)
        accuracies.append(accuracy)
        remaining_rate.append(i)

    sparsity = simpson_rule(accuracies)

    return sparsity, accuracies, remaining_rate
