import yaml
import os
import json
import matplotlib.pyplot as plt

def load_hyperparameters(file_name="hyperparameters.yaml"):
    try:
        with open(file_name, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        exit(1)
    return config

def save_stats(stats, stats_dir="stats/train", file_name="train_stats_lisa.json"):
    os.makedirs(stats_dir, exist_ok=True)
    path = os.path.join(stats_dir, file_name)
    with open(path, "w") as f:
        json.dump(stats, f, indent=4)

def plot(stats_dir="stats/train", plot_name="train_losses", plot_dir="plots"):
    os.makedirs(plot_dir, exist_ok=True)

    strategies = ["lisa", "intra_label", "intra_domain", "standard", "none"]
    plt.figure(figsize=(10, 6), dpi=300)

    for strategy in strategies:
        path = os.path.join(stats_dir, f"train_stats_{strategy}.json")
        with open(path, "r") as f:
            stats = json.load(f)

        for key, values in stats.items():
            plt.plot(values, label=f"{strategy}_{key}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Across Strategies")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(plot_dir, f"{plot_name}.png")
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")