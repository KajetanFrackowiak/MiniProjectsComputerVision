import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

class ResidualCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.b2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        out = self.bn1(x)
        out = swish(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = swish(out)
        out = self.conv2(out)

        skip = self.skip(x)
        return out + skip
    

class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim_per_group=16, num_groups=4):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.ModuleDict()
        self.fc_mu = nn.ModuleDict()
        self.fc_logvar = nn.ModuleDict()

        channels = 32
        for i in range(num_groups):
            out_channels = channels * 2 # increase channels per group
            self.res_blocks.append(ResidualCell(channels, out_channels))
            self.fc_mu.append(nn.Linear(out_channels * 8 * 8, latent_dim_per_group))
            self.fc_logvar.append(nn.Linear(out_channels * 8 * 8, latent_dim_per_group))
            channels = out_channels
    
    def forward(self, x):
        x = self.initial_conv(x)
        latent_groups = []

        for res_block, fc_mu, fc_logvar in zip(self.res_blocks, self.fc_mu, self.fc_logvar):
            x = res_block(x)
            x = F.avg_pool2d(x, kernel_size=2)
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            mu = fc_mu(x_flat)
            logvar = fc_logvar(x_flat)
            latent_groups.append((mu, logvar))
        
        return latent_groups