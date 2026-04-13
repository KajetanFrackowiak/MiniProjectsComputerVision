import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(json_path, output_path):
    with open(json_path) as f:
        history = json.load(f)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "DejaVu Serif"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )

    fig, ax = plt.subplots(figsize=(5, 3.5))

    steps = history["step"]
    train_loss = history["train/loss_avg"]
    eval_loss = history["eval/loss_avg"]

    ax.plot(steps, train_loss, label="Training", color="#1c2d5a", lw=1.5)
    ax.plot(steps, eval_loss, label="Evaluation", color="#e0422f", lw=1.2, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel(r"Flow Matching Loss ($\mathcal{L}$)")

    ax.locator_params(axis="y", nbins=5)
    ax.locator_params(axis="x", nbins=8)

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    print(f"Clean plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument(
        "--json_path", type=str, default="../results/history_cifar10_seed_2268134838_final.json"
    )
    parser.add_argument(
        "--output_path", type=str, default="../plots/cifar10_training_plot_seed_2268134838.png"
    )
    args = parser.parse_args()
    os.makedirs("../plots", exist_ok=True)
    plot(args.json_path, args.output_path)
