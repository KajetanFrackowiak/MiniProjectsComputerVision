import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.conv2d(x, self.conv1.weight, self.conv1.bias, stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(x, self.conv2.weight, self.conv2.bias, stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(x, self.conv3.weight, self.conv3.bias, stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(x, self.conv4.weight, self.conv4.bias, stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        return x

    def functional_forward(self, x, weights):
        x = F.conv2d(x, weights["conv1.weight"], weights["conv1.bias"], stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 28x28 -> 14x14

        x = F.conv2d(x, weights["conv2.weight"], weights["conv2.bias"], stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 14x14 -> 7x7

        x = F.conv2d(x, weights["conv3.weight"], weights["conv3.bias"], stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 7x7 -> 3x3

        x = F.conv2d(x, weights["conv4.weight"], weights["conv4.bias"], stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 3x3 -> 1x1

        x = x.view(x.size(0), -1) # [batch_size, hidden_channels*1*1]
        return x

