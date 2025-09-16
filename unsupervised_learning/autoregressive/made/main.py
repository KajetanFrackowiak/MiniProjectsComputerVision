import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from data import load_data
from made import MADE
from training import train


def load_hyperparameters():
    with open("hyperparameters.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


def plot(losses, training=True, file_name="training"):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 8))

    # normalize losses to a plain Python list of floats to avoid matplotlib
    # attempting to convert CUDA tensors to numpy arrays
    proc_losses = []
    for v in losses:
        if isinstance(v, torch.Tensor):
            proc_losses.append(float(v.detach().cpu().item()))
        else:
            proc_losses.append(float(v))

    plt.plot(proc_losses, label="Mean Losses", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if training:
        plt.title("MADE Training Loss")
    else:
        plt.title("MADE Testing Loss")

    plt.tight_layout()
    plt.savefig(f"plots/{file_name}.png")
    plt.show()
    plt.close()


def main():
    trainloader, testloader = load_data()
    config = load_hyperparameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    made = MADE(config["input_dim"], config["output_dim"]).to(device)
    optimizer = optim.Adam(made.parameters(), lr=config["learning_rate"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-5
    )

    avg_losses = train(made, trainloader, optimizer, scheduler, config["epochs"], device)
    plot(avg_losses)


if __name__ == "__main__":
    main()
