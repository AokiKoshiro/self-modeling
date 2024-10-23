import random

import numpy as np
import torch
import yaml


def get_device(device_preference: str = "cuda") -> torch.device:
    return torch.device(device_preference if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def gen_key(hidden_size: int, aw: float, l2_lambda: float) -> str:
    return f"hidden_{hidden_size}_aw_{aw}_l2_{l2_lambda}"
