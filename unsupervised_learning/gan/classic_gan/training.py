import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, train_loader, latent_dim, generator, discriminator, 
                 gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler, batch_size, device, 
                 checkpoint_dir="checkpoints", checkpoint_interval=10, epochs=100):
        self.train_loader = train_loader
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.epochs=epochs
        

    def train_step(self, real_images):
        self.generator.train()
        self.discriminator.train()

        real_images = real_images.view(real_images.size(0), -1).to(self.device)
        batch_size = real_images.size(0)

        # Train Discriminator
        self.disc_optimizer.zero_grad()

        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_output = self.discriminator(real_images)
        real_loss = F.binary_cross_entropy(real_output, real_labels)

        z = torch.randn(batch_size, 100).to(self.device)
        fake_images = self.generator(z)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_images.detach())
        fake_loss = F.binary_cross_entropy(fake_output, fake_labels)

        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Train Generator
        self.gen_optimizer.zero_grad()

        z = torch.randn(batch_size, 100).to(self.device)
        fake_images = self.generator(z)
        output = self.discriminator(fake_images)
        gen_loss = F.binary_cross_entropy(output, real_labels)

        gen_loss.backward()
        self.gen_optimizer.step()
        
        return {"disc_loss": disc_loss.item(), "gen_loss": gen_loss.item()}

    def train_batch(self):
        batch_stats = {
            "avg_disc_loss": 0,
            "avg_gen_loss": 0,
        }

        for i, batch in enumerate(tqdm(self.train_loader)):
            real_images, _ = batch
            real_images = real_images.to(self.device)
            losses = self.train_step(real_images)
            disc_loss = losses["disc_loss"]
            gen_loss = losses["gen_loss"]

            batch_stats["avg_disc_loss"] += disc_loss
            batch_stats["avg_gen_loss"] += gen_loss

            self.gen_scheduler.step()
            self.disc_scheduler.step()

        for key in batch_stats:
            batch_stats[key] /= len(self.train_loader) 
        
        return batch_stats

    def train(self):
        train_stats = {
            "avg_disc_losses": [],
            "avg_gen_losses": [],
        }

        for epoch in range(self.epochs):
            batch_stats = self.train_batch()
            train_stats["avg_disc_losses"].append(batch_stats["avg_disc_loss"])
            train_stats["avg_gen_losses"].append(batch_stats["avg_gen_loss"])

            print(f"Epoch: {epoch+1}/{self.epochs}, Avg Disc Loss: {batch_stats['avg_disc_loss']:.4f}, Avg Gen Loss: {batch_stats['avg_gen_loss']:.4f}")

            log_data = {
                "epoch": epoch+1,
                **batch_stats,
            }

            wandb.log(log_data)
            
            if (epoch + 1) % 10 == 0:
                z = torch.randn(64, self.latent_dim).to(self.device)
                fake_images = self.generator(z).view(-1, 1, 28, 28)
                wandb.log({"Generated Images": [wandb.Image(img) for img in fake_images]})
                
            if (epoch + 1) % self.checkpoint_interval == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                torch.save({
                    "discriminator": self.discriminator.state_dict(),
                    "generator": self.generator.state_dict(),
                    "disc_optimizer": self.disc_optimizer.state_dict(),
                    "gen_optimizer": self.gen_optimizer.state_dict(),
                    "epoch": epoch+1
                }, f"{self.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")

        return train_stats