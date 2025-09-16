import matplotlib.pyplot as plt
import os
import yaml


def load_hyperparameters(file_name="hyperparameters.yaml"):
    try:
        with open(file_name, "r") as f:
            conifg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        exit(1)
    return conifg

def reconstructions_grid(real_images, recon_images, rows=2, cols=8, train=True, save_path=None):
    if train:
        os.makedirs("plots/recon/train")
    else:
        os.makedirs("plots/recon/test")

    n_images = rows * cols // 2
    real_images = real_images[:n_images].cpu()
    recon_images = recon_images[:n_images].cpu()

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    for i in range(cols):
        axes[0, i].imshow(real_images[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Real")
        
        axes[1, i].imshow(recon_images[i].permute(1, 2, 0) * 0.5 + 0.5)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("VAE Recon")
    
    plt.subplots_adjust(hspace=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

def plot(vq_losses, recon_losses, total_losses, train=True):
    os.makedirs("plots/loss", exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.plot(vq_losses, color="red", linestyle="--", label="VQ Loss")
    plt.plot(recon_losses, color="blue", linestyle="-", label="Recon Loss")
    plt.plot(total_losses, color="green", linestyle=":", label="Total Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    if train:
        plt.title("Training")
    else:
        plt.title("Testing")
    plt.legend()
    plt.tight_layout()
    
    if train:
        plt.savefig("plots/loss/training.png")
    else:
        plt.savefig("plots/loss/testing.png")