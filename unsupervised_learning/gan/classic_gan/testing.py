import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from pytorch_fid import fid_score
from tqdm import tqdm
import wandb
import os

def save_generated_images(generator, latent_dim, save_dir, device, n_samples=5000):
        os.makedirs(save_dir, exist_ok=True)
        batch_size = 64
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_images = generator(z).view(-1, 1, 28,28)
                for j, img in enumerate(fake_images):
                    vutils.save_image(img, f"{save_dir}/img_{i+j}.png", normalize=True)

def save_real_images(self, save_dir="data/real_images"):
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    with torch.no_grad():
        for images, _ in self.test_loader:
            for img in images:
                vutils.save_image(img, f"{save_dir}/img_{idx}.png", normalize=True)
                idx += 1
    return save_dir

class Tester:
    def __init__(self, test_loader, generator, discriminator, latent_dim, device, n_samples):
        self.test_loader = test_loader
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.device = device
        self.n_samples = n_samples

        self.discriminator.eval()
        self.generator.eval()
    
    def save_real_images(self, save_dir="data/real_images"):
        os.makedirs(save_dir, exist_ok=True)
        idx = 0
        with torch.no_grad():
            for images, _ in self.test_loader:
                for img in images:
                    vutils.save_image(img, f"{save_dir}/img_{idx}.png", normalize=True)
                    idx += 1
        return save_dir

    def test_batch(self):
        batch_stats = {"avg_disc_loss": 0}
        num_batches = len(self.test_loader)

        with torch.no_grad():
            for real_images, _ in tqdm(self.test_loader):
                batch_size = real_images.size(0)
                real_images = real_images.view(batch_size, -1).to(self.device)

                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                real_output = self.discriminator(real_images)
                real_loss = F.binary_cross_entropy(real_output, real_labels)
                
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(z)
                fake_output = self.discriminator(fake_images)
                fake_loss = F.binary_cross_entropy(fake_output, fake_labels)

                disc_loss = real_loss + fake_loss
                batch_stats["avg_disc_loss"] += disc_loss.item()
            
        batch_stats["avg_disc_loss"] /= num_batches
        return batch_stats

    def evaluate_fid(self, n_samples=5000, real_images_dir="data/real_images", fake_images_dir="data/generated_images"):
        save_generated_images(self.generator, self.latent_dim, fake_images_dir, self.device, n_samples)
        self.save_real_images(save_dir=real_images_dir)
        fid_value = fid_score.calculate_fid_given_paths(
            [real_images_dir, fake_images_dir],
            batch_size=64,
            device=self.device,
            dims=2048
        )
        return fid_value

    def test(self):
        test_stats = self.test_batch()
        fid_value = self.evaluate_fid(n_samples=self.n_samples)
        
        print(f"Avg Discriminator Loss on Test Set: {test_stats['avg_disc_loss']:.4f}")
        print(f"FID value: {fid_value:.4f}")

        wandb.log(test_stats)
        wandb.log({"FID": fid_value})
