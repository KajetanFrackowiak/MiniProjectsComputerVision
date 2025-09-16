import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=4, stride=2, padding=1) # (batch_size, 16, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1) # (batch_size, 15, 64, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0) # (batch_size, 13, 128, 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # (batch_size, 11, 256, 256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0) # (batch_size, 9, 512, 512)
        self.fc = nn.Linear(512 * 9 * 9, 2 * latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1) # (batch_size, 512 * 9 * 9)
        x = self.fc(x)

        mu, logvar = x.chunk(2, dim=1)

        return mu, logvar



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: K embeddings of size D
        self.embedding = nn.Embedding(num_embeddings, embedding_dim) # (K, D)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings) # (K, D)

    def forward(self, z):
        """
        z: (batch, channels, height, width)
        """
        z_perm = z.permute(0, 2, 3, 1).contiguous() # (batch, height, width, channels)
        z_flat = z_perm.view(-1, self.embedding_dim) # (batch * height * width, channels)

        # Compute distances between z and embeddings
        # || z - e ||^2 = ||z||^2 - 2 * z.e + ||e||^2
        distances = (
            z_flat.pow(2).sum(1, keepdim=True) - 2 * z_flat @ self.embedding.weight.t() + self.embedding.weight.pow(2).sum(1)) # (N, K)
        
        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (N, 1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device) # (N, K)
        encodings.scatter_(1, encoding_indices, 1) # (N, K)

        # Quantize: replace z with nearest embeddings
        quantized = encodings @ self.embedding.weight # (N, D)
        quantized = quantized.view(z_perm.shape) # (batch, height, width, channels)
        
        # Loss
        codebook_loss = F.mse_loss(quantized.detach(), z_perm) # (batch, height, width, channels)
        commitment_loss = self.commitment_cost *  F.mse_loss(quantized, z_perm.detach()) # (batch, height, width, channels)
        loss = codebook_loss + commitment_loss

        # Straight-through estimator
        quantized = z_perm + (quantized - z_perm).detach() # (batch, height, width, channels)
        quantized = quantized.permute(0, 3, 1, 2).contiguous() # (batch, channels, height, width)

        return quantized, loss, encoding_indices
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 9 * 9)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, output_dim, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(x.size(0), 512, 9, 9)
        
        x = self.deconv1(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = F.relu(x)

        x = self.deconv3(x)
        x = F.relu(x)

        x = self.deconv4(x)
        x = F.relu(x)

        x = self.deconv5(x)
        return x