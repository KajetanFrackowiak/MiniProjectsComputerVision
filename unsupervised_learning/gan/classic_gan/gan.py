import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=128, img_dim=28*28):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, img_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.tanh(x)

        return x
    

class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28, hidden_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(img_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.linear2(x)
        x = F.sigmoid(x)
        return x
    

    