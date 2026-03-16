import json

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def save_stats(stats: dict, save_path: str) -> None:
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)


def load_stats(stats_path: str) -> dict:
    with open(stats_path) as f:
        stats = json.load(f)
    return stats
