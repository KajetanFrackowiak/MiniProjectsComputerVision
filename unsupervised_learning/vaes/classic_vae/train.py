import torch
from tqdm import tqdm

from vae import  reparameterize, vae_loss

class Trainer:

    def __init__(self, encoder, decoder, train_loader, optimizer, device):
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.num_samples = 0

        self.losses = []
        self.recon_losses = []
        self.kl_losses = []

    def train_one_epoch(self):
        self.num_samples = 0

        self.encoder.train()
        self.decoder.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        for batch in tqdm(self.train_loader, desc="train"):
            imgs = batch["pixel_values"].to(self.device).float()
            self.num_samples += imgs.size(0)

            self.optimizer.zero_grad()
            mu, logvar = self.encoder(imgs)
            z = reparameterize(mu, logvar)
            x_recon = self.decoder(z)
            loss, recon_loss, kl_loss = vae_loss(x_recon, imgs, mu, logvar)
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        return total_loss, total_recon, total_kl

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            total_loss, total_recon, total_kl = self.train_one_epoch()
            avg_loss = total_loss / self.num_samples
            avg_recon_loss = total_recon / self.num_samples
            avg_kl_loss = total_kl / self.num_samples

            self.losses.append(avg_loss)
            self.recon_losses.append(avg_recon_loss) 
            self.kl_losses.append(avg_kl_loss)

            print(
                f"Epoch {epoch}: loss={avg_loss:.4f}, recon={avg_recon_loss:.4f}, kl={avg_kl_loss:.4f}"
            )
        
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, f"vae_checkpoint_epoch_{epochs}.pt")

        return self.losses, self.recon_losses, self.kl_losses
