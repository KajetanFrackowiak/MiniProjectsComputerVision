import json
import os
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

    epochs = history["epoch"]

    ax.plot(epochs, history["train_loss"], label="Training", color="#1c2d5a", lw=1.5)
    ax.plot(epochs, history["val_loss"], label="Validation", color="#e0422f", lw=1.2, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"VAE Loss ($\mathcal{L}$)")

    ax.locator_params(axis="y", nbins=5)
    ax.locator_params(axis="x", nbins=10)

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Clean plot saved to {output_path}")


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot("results/training_history.json", "plots/training_plot.pdf")
