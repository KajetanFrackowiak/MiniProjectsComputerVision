import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Residual Blcok (No Bottleneck)"""
    expansion = 1  # Output channels remain the same

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()  # Default: identity shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        x = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity # skip connection
        return nn.ReLU(inplace=True)(x)
    

class ResNet34_CIFAR10(nn.Module):
    """ResNet-34 adapted for CIFAR-10 (32x32 input)"""
    def __init__(self, num_classes=10):
        super(ResNet34_CIFAR10, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Changed kernel & stride
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # REMOVED MaxPool
        
        self.layer1 = self._make_layer(64, 3)  # 3 blocks
        self.layer2 = self._make_layer(128, 4, stride=2)  # 4 blocks
        self.layer3 = self._make_layer(256, 6, stride=2)  # 6 blocks
        self.layer4 = self._make_layer(512, 3, stride=2)  # 3 blocks

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride=1):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x