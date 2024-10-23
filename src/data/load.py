import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common.tools import load_config


def get_data_loaders(config_path: str = "config.yaml") -> tuple:
    config = load_config(config_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["data"]["normalize"]["mean"],
                std=config["data"]["normalize"]["std"],
            ),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size_train"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["data"]["batch_size_test"], shuffle=False
    )

    return train_loader, test_loader
