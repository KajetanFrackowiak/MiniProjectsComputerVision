import torch.nn as nn
from torchvision import models

class LISA(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(LISA, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
    
