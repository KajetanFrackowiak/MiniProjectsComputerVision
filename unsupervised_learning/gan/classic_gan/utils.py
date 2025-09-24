import matplotlib.pyplot as plt
import yaml
import os
import numpy as np

def load_hyperparameters(file_name="hyperparameters.yaml"):
    try:
        with open(file_name, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        exit(1)
    return config

def plot(stats, dir_name=None, file_name=None, smoothing_window=5):
    os.makedirs(dir_name, exist_ok=True)

    def smooth(x, w):
        return np.convolve(x, np.ones(w)/w, mode="valid")
    
    disc_smooth = smooth(stats["avg_disc_losses"], smoothing_window)
    gen_smooth = smooth(stats["avg_gen_losses"], smoothing_window)
    epochs = np.arange(1, len(disc_smooth) + 1)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(epochs, disc_smooth, label="Discriminator Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, gen_smooth, label="Generator Loss", color="#ff7f0e", linewidth=2)

    plt.grid(True, linestyle="--", alpha=0.3)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Loss", fontsize=12, fontweight="bold")
    
    plt.title("GAN Training Losses", fontsize=14, fontweight="bold")
   
    plt.legend(fontsize=12)

    plt.tight_layout()

    if file_name:
        plt.savefig(f"{dir_name}/{file_name}", dpi=600)
    
    plt.show()
    