from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.tools import gen_key


def create_and_save_plot(
    x_data, y_data, ci_data, labels, x_label, y_label, title, filename
):
    plt.figure(figsize=(10, 6))

    y_data = np.array(y_data)
    ci_data = np.array(ci_data)
    for i, label in enumerate(labels):
        plt.plot(x_data, y_data[i], label=label)
        plt.fill_between(
            x_data, y_data[i] - ci_data[i], y_data[i] + ci_data[i], alpha=0.1
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = Path("results") / "figures" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_weight_std_over_time(
    results, hidden_size, aw_values, l2_lambdas, num_epochs
):
    epochs = np.arange(1, num_epochs + 1)

    # Self-modeling
    y_data_sm = [
        results[gen_key(hidden_size, aw, 0)]["weight_std_dev"]
        for aw in aw_values
        if aw > 0
    ]
    ci_data_sm = [
        results[gen_key(hidden_size, aw, 0)]["ci_weight_std_dev"]
        for aw in aw_values
        if aw > 0
    ]
    labels_sm = [f"SM (AW={aw})" for aw in aw_values if aw > 0]

    # L2 regularization
    y_data_l2 = [
        results[gen_key(hidden_size, 0, l2)]["weight_std_dev"]
        for l2 in l2_lambdas
        if l2 > 0
    ]
    ci_data_l2 = [
        results[gen_key(hidden_size, 0, l2)]["ci_weight_std_dev"]
        for l2 in l2_lambdas
        if l2 > 0
    ]
    labels_l2 = [f"L2 (λ={l2})" for l2 in l2_lambdas if l2 > 0]

    # Baseline (no regularization)
    y_data_base = [results[gen_key(hidden_size, 0, 0)]["weight_std_dev"]]
    ci_data_base = [results[gen_key(hidden_size, 0, 0)]["ci_weight_std_dev"]]
    labels_base = ["Baseline"]

    y_data = y_data_sm + y_data_l2 + y_data_base
    ci_data = ci_data_sm + ci_data_l2 + ci_data_base
    labels = labels_sm + labels_l2 + labels_base

    create_and_save_plot(
        epochs,
        y_data,
        ci_data,
        labels,
        "Epochs",
        "Standard Deviation of Weight Distribution",
        f"Weight Distribution Change Over Time (Hidden Size={hidden_size})",
        f"weight_std_over_time_hidden_{hidden_size}.png",
    )


def plot_final_weight_std(results, hidden_sizes, aw_values, l2_lambdas):
    # Self-modeling
    y_data_sm = [
        [results[gen_key(hs, aw, 0)]["weight_std_dev"][-1] for hs in hidden_sizes]
        for aw in aw_values
        if aw > 0
    ]
    ci_data_sm = [
        [results[gen_key(hs, aw, 0)]["ci_weight_std_dev"][-1] for hs in hidden_sizes]
        for aw in aw_values
        if aw > 0
    ]
    labels_sm = [f"SM (AW={aw})" for aw in aw_values if aw > 0]

    # L2 regularization
    y_data_l2 = [
        [results[gen_key(hs, 0, l2)]["weight_std_dev"][-1] for hs in hidden_sizes]
        for l2 in l2_lambdas
        if l2 > 0
    ]
    ci_data_l2 = [
        [results[gen_key(hs, 0, l2)]["ci_weight_std_dev"][-1] for hs in hidden_sizes]
        for l2 in l2_lambdas
        if l2 > 0
    ]
    labels_l2 = [f"L2 (λ={l2})" for l2 in l2_lambdas if l2 > 0]

    # Baseline (no regularization)
    y_data_base = [
        [results[gen_key(hs, 0, 0)]["weight_std_dev"][-1] for hs in hidden_sizes]
    ]
    ci_data_base = [
        [results[gen_key(hs, 0, 0)]["ci_weight_std_dev"][-1] for hs in hidden_sizes]
    ]
    labels_base = ["Baseline"]

    y_data = y_data_sm + y_data_l2 + y_data_base
    ci_data = ci_data_sm + ci_data_l2 + ci_data_base
    labels = labels_sm + labels_l2 + labels_base

    create_and_save_plot(
        hidden_sizes,
        y_data,
        ci_data,
        labels,
        "Hidden Layer Size",
        "Standard Deviation of Weight Distribution at Final Epoch",
        "Comparison of Weight Distribution Standard Deviation at Final Epoch",
        "final_weight_std.png",
    )


def plot_test_accuracy(results, hidden_sizes, aw_values, l2_lambdas):
    # Self-modeling
    y_data_sm = [
        [results[gen_key(hs, aw, 0)]["accuracy"][0] for hs in hidden_sizes]
        for aw in aw_values
        if aw > 0
    ]
    ci_data_sm = [
        [results[gen_key(hs, aw, 0)]["accuracy"][1] for hs in hidden_sizes]
        for aw in aw_values
        if aw > 0
    ]
    labels_sm = [f"SM (AW={aw})" for aw in aw_values if aw > 0]

    # L2 regularization
    y_data_l2 = [
        [results[gen_key(hs, 0, l2)]["accuracy"][0] for hs in hidden_sizes]
        for l2 in l2_lambdas
        if l2 > 0
    ]
    ci_data_l2 = [
        [results[gen_key(hs, 0, l2)]["accuracy"][1] for hs in hidden_sizes]
        for l2 in l2_lambdas
        if l2 > 0
    ]
    labels_l2 = [f"L2 (λ={l2})" for l2 in l2_lambdas if l2 > 0]

    # Baseline (no regularization)
    y_data_base = [[results[gen_key(hs, 0, 0)]["accuracy"][0] for hs in hidden_sizes]]
    ci_data_base = [[results[gen_key(hs, 0, 0)]["accuracy"][1] for hs in hidden_sizes]]
    labels_base = ["Baseline"]

    y_data = y_data_sm + y_data_l2 + y_data_base
    ci_data = ci_data_sm + ci_data_l2 + ci_data_base
    labels = labels_sm + labels_l2 + labels_base

    create_and_save_plot(
        hidden_sizes,
        y_data,
        ci_data,
        labels,
        "Hidden Layer Size",
        "Test Accuracy",
        "Comparison of Test Accuracy for Different Hidden Layer Sizes",
        "test_accuracy.png",
    )


def plot_sparsity(results, hidden_sizes, aw_values, l2_lambdas):
    # Self-modeling
    y_data_sm = [
        [results[gen_key(hs, aw, 0)]["sparsity"][0] for hs in hidden_sizes]
        for aw in aw_values
        if aw > 0
    ]
    ci_data_sm = [
        [results[gen_key(hs, aw, 0)]["sparsity"][1] for hs in hidden_sizes]
        for aw in aw_values
        if aw > 0
    ]
    labels_sm = [f"SM (AW={aw})" for aw in aw_values if aw > 0]

    # L2 regularization
    y_data_l2 = [
        [results[gen_key(hs, 0, l2)]["sparsity"][0] for hs in hidden_sizes]
        for l2 in l2_lambdas
        if l2 > 0
    ]
    ci_data_l2 = [
        [results[gen_key(hs, 0, l2)]["sparsity"][1] for hs in hidden_sizes]
        for l2 in l2_lambdas
        if l2 > 0
    ]
    labels_l2 = [f"L2 (λ={l2})" for l2 in l2_lambdas if l2 > 0]

    # Baseline (no regularization)
    y_data_base = [[results[gen_key(hs, 0, 0)]["sparsity"][0] for hs in hidden_sizes]]
    ci_data_base = [[results[gen_key(hs, 0, 0)]["sparsity"][1] for hs in hidden_sizes]]
    labels_base = ["Baseline"]

    y_data = y_data_sm + y_data_l2 + y_data_base
    ci_data = ci_data_sm + ci_data_l2 + ci_data_base
    labels = labels_sm + labels_l2 + labels_base

    create_and_save_plot(
        hidden_sizes,
        y_data,
        ci_data,
        labels,
        "Hidden Layer Size",
        "Sparsity",
        "Comparison of Sparsity for Different Hidden Layer Sizes",
        "sparsity.png",
    )
