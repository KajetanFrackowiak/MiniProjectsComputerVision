import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml
import argparse
import os

from tqdm import tqdm
from vae import Encoder, Decoder, reparameterize, vae_loss
from data import load_data
from train import Trainer


def load_hyperparamters():
    try:
        with open("hyperparameters.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: hyperparameters.yaml not found.")
        exit(1)
    return config


def plot(losses, recon_losses, kl_losses, training=True, file_name="vae_training"):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.plot(losses, label="Total loss", linewidth=2)
    plt.plot(recon_losses, label="Reconstruction Loss", linestyle="--")
    plt.plot(kl_losses, label="KL Divergence", linestyle=":")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if training:
        plt.title("VAE Training Loss Components")
    else:
        plt.title("VAE Testing Loss Components") 
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.savefig(f"plots/{file_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_hyperparamters()
    latent_dim = config["latent_dim"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    train_loader, _ = load_data()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params=params, lr=learning_rate)
    trainer = Trainer(encoder, decoder, train_loader, optimizer, device)

    losses, recon_losses, kl_losses = trainer.train(epochs)
    plot(losses, recon_losses, kl_losses, training=True, file_name="vae_training")

def inference(config, path="vae_checkpoint_epoch_10"):
    latent_dim = config["latent_dim"]

    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)

    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    _, test_loader = load_data()
    
    losses = []
    recon_losses = []
    kl_losses = []

    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        for batch in tqdm(test_loader, desc="test"):
            imgs = batch["pixel_values"]
            
            mu, logvar = encoder(imgs)
            z = reparameterize(mu, logvar)
            x_recon = decoder(z)
            loss, recon_loss, kl_loss = vae_loss(x_recon, imgs, mu, logvar)

            losses.append(loss/len(test_loader))
            recon_losses.append(recon_loss/len(test_loader))
            kl_losses.append(kl_loss/len(test_loader))
        
        plot(losses, recon_losses, kl_losses, training=False, file_name="vae_testing")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="vae_checkpoint_epoch_10.pt")
    args = parser.parse_args()

    config = load_hyperparamters()

    # training(config)
    inference(config, args.checkpoint)
