import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_dim, 32, kernel_size=4, stride=2, padding=1
        )  # (batch_size, 32, 16, 16)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # (batch_size, 64, 8, 8)
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=1, padding=0
        )  # (batch_size, 128, 5, 5)
        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=2, stride=1, padding=0
        )  # (batch_size, 256, 4, 4)
        self.conv5 = nn.Conv2d(
            256, 512, kernel_size=2, stride=1, padding=0
        )  # (batch_size, 512, 3, 3)
        self.fc = nn.Linear(512 * 3 * 3, 2 * latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # (batch_size, 512*3*3)
        x = self.fc(x)  # (batch_size, 2*latent_dim)
        mu, logvar = x.chunk(
            2, dim=1
        )  # (batch_size, latent_dim), (batch_size, latent_dim)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 3 * 3)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(
            32, input_dim, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), 512, 3, 3)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        return x


class VaeWithFlows(nn.Module):
    def __init__(self, encoder, decoder, flows):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.flows = flows

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z0 = self.reparameterize(mu, logvar)
        log_det_sum = 0.0

        z = z0
        log_det_sum = torch.zeros(z.size(0), device=z.device)
        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            log_det_sum += log_det_jacobian.squeeze()

        recon = self.decoder(z)
        return recon, mu, logvar, log_det_sum


def vae_flow_loss(recon, x, mu, logvar, log_det_sum):
    # MSE loss over each pixel
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

    # KL divergence between q(z0|x) and p(z0) = N(0, I)
    kl_base = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # Adjust KL with log-det-Jacobian from flows
    kl_loss = kl_base - log_det_sum
    kl_loss = kl_loss.mean()
    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss
