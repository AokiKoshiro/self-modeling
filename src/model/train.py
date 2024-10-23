import itertools
import logging
from pathlib import Path

import numpy as np
import scipy
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.common.tools import gen_key, get_device, load_config, set_seed
from src.data.load import get_data_loaders
from src.model.evaluate import evaluate
from src.model.measure_complexity import measure_sparsity, measure_weight_distribution
from src.model.model import SelfModelingMLP
from src.visualization.visualize import (
    plot_final_weight_std,
    plot_sparsity,
    plot_test_accuracy,
    plot_weight_std_over_time,
)


def setup_logging(log_path: str = "logs/training.log"):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def train_epoch(model, device, train_loader, optimizer, aw, l2_lambda):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, self_pred, hidden = model(data)
        loss_class = F.cross_entropy(logits, target)
        loss_self = F.mse_loss(self_pred, hidden.detach())
        loss = loss_class + aw * loss_self
        loss += l2_lambda * model.get_mean_l2_norm()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def execute_run(
    config, device, train_loader, test_loader, hidden_size, aw, l2_lambda, run, run_seed
):
    set_seed(run_seed)
    logging.info(f"Run {run+1}/{config['train']['num_runs']}")

    model = SelfModelingMLP(
        input_size=config["model"]["input_size"],
        hidden_size=hidden_size,
        num_classes=config["model"]["num_classes"],
        aw=aw,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    weight_std_devs = []

    with tqdm(
        range(1, config["train"]["num_epochs"] + 1),
        desc="Epoch",
        position=1,
        leave=False,
    ) as pbar:
        for epoch in pbar:
            train_loss = train_epoch(
                model, device, train_loader, optimizer, aw, l2_lambda
            )
            std_dev = measure_weight_distribution(model)
            weight_std_devs.append(std_dev)
            pbar.set_postfix({"Loss": f"{train_loss:.4f}", "Std Dev": f"{std_dev:.4f}"})

    accuracy = evaluate(model, device, test_loader)
    sparsity, _, _ = measure_sparsity(
        model, config["measure_complexity"]["pruning_interval"]
    )
    logging.info(
        f"Run {run+1} - Test accuracy: {accuracy:.4f}, Sparsity: {sparsity:.4f}"
    )

    return weight_std_devs, accuracy, sparsity


def train():
    setup_logging()

    config = load_config()

    set_seed(config["seed"])
    logging.info(f"Random seed set to {config['seed']}")

    device = get_device(config["device"])
    logging.info(f"Device: {device}")

    train_loader, test_loader = get_data_loaders()

    hidden_sizes = config["model"]["hidden_sizes"]
    aw_values = config["train"]["aw_values"]
    l2_lambdas = config["train"]["l2_lambdas"]
    num_epochs = config["train"]["num_epochs"]
    num_runs = config["train"]["num_runs"]

    results = {}

    for hidden_size, aw, l2_lambda in itertools.product(
        hidden_sizes, aw_values, l2_lambdas
    ):
        if aw > 0 and l2_lambda > 0:
            continue
        key = gen_key(hidden_size, aw, l2_lambda)
        logging.info(
            f"Training: hidden_size={hidden_size}, aw={aw}, l2_lambda={l2_lambda}"
        )
        weight_distributions = []
        accuracies = []
        sparsities = []
        for run in tqdm(range(num_runs), desc=f"Run {key}", position=0):
            run_seed = config["seed"] + run
            weight_std_devs, accuracy, sparsity = execute_run(
                config,
                device,
                train_loader,
                test_loader,
                hidden_size,
                aw,
                l2_lambda,
                run,
                run_seed,
            )
            weight_distributions.append(weight_std_devs)
            accuracies.append(accuracy)
            sparsities.append(sparsity)

        t_value = scipy.stats.t.ppf(0.975, df=num_runs - 1)
        mean_weight_std = np.mean(weight_distributions, axis=0)
        ci_weight_std = (
            t_value * np.std(weight_distributions, axis=0, ddof=1) / np.sqrt(num_runs)
        )
        mean_accuracy = np.mean(accuracies)
        ci_accuracy = t_value * np.std(accuracies, ddof=1) / np.sqrt(num_runs)
        mean_sparsity = np.mean(sparsities)
        ci_sparsity = t_value * np.std(sparsities, ddof=1) / np.sqrt(num_runs)

        results[key] = {
            "weight_std_dev": mean_weight_std,
            "ci_weight_std_dev": ci_weight_std,
            "accuracy": (mean_accuracy, ci_accuracy),
            "sparsity": (mean_sparsity, ci_sparsity),
        }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.npy"
    np.save(results_file, results)

    plot_weight_std_over_time(
        results, hidden_sizes[-1], aw_values, l2_lambdas, num_epochs
    )
    plot_final_weight_std(results, hidden_sizes, aw_values, l2_lambdas)
    plot_test_accuracy(results, hidden_sizes, aw_values, l2_lambdas)
    plot_sparsity(results, hidden_sizes, aw_values, l2_lambdas)

    logging.info("Training finished.")


if __name__ == "__main__":
    train()
