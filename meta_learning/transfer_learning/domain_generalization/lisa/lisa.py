import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class LISA(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super(LISA, self).__init__()
        self.backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):  # everything except classifier
                    param.requires_grad = False

    def unfreeze_backbone_layers(self, layer_names=None):
        if layer_names is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in self.backbone.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True

    def freeze_backbone_layers(self, layer_names=None):
        if layer_names is None:
            # Freeze all backbone layers except fc
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
        else:
            # Freeze specific layers
            for name, param in self.backbone.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
