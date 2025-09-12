import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1) # (batch_size, 32, 16, 16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # (batch_size, 64, 8, 8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0) # (batch_size, 128, 5, 5)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0) # (batch_size, 256, 4, 4)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0) # (batch_size, 512, 3, 3)
        self.fc = nn.Linear(512 * 3 * 3, 2 * latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar


def reparameterize(mu, logvar):
    """Reparameterization: sample z ~ N(mu, sigma^2) from mu and log-variance."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 3 * 3)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, input_dim, kernel_size=4, stride=2, padding=1)

    

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), 512, 3, 3)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        return x


def vae_loss(x_recon, x, mu, logvar):
    # mse_loss pixel by pixel
    recon_loss = F.mse_loss(x_recon, x, reduction="sum")
    # KL(Normal(mu, sigma^2 || Normal(0, 1)))
    kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss - kl_loss
    return loss, recon_loss, kl_loss
