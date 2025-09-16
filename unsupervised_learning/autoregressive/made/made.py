import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        hidden_layers = [512, 512, 512]
        masks = self.create_masks(input_dim, hidden_layers)

        self.layer1 = MaskedLinear(input_dim, 512, mask=masks[0])
        self.layer2 = MaskedLinear(512, 512, mask=masks[1])
        self.layer3 = MaskedLinear(512, 512, mask=masks[2])
        self.layer4 = MaskedLinear(512, output_dim, mask=masks[3])
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

    def create_masks(self, input_dim, hidden_dims):
        D = input_dim
        masks = []

        degrees = []
        degrees.append(torch.arange(1, D + 1))  # input layer
        for h in hidden_dims:
            degrees.append(torch.randint(1, D + 1, (h,)))  # hidden layers
        degrees.append(torch.arange(1, D + 1))  # output layer

        for l in range(len(degrees) - 1):
            mask = (degrees[l + 1].unsqueeze(-1) >= degrees[l].unsqueeze(0)).float()
            masks.append(mask)

        return masks
