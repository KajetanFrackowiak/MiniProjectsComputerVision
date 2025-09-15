import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    
def grl(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(48)
    
    def forward(self, x):
        x = self.conv1(x) # (batch_size, 32, 24, 24)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # (batch_size, 32, 12, 12)

        x = self.conv2(x) # (batch_size, 48, 8, 8)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) # (batch_size, 48, 4, 4)

        x = x.view(x.size(0), -1) # (batch_size, 48*4*4)
        return x


class LabelPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 100)
        self.linear2 = nn.Linear(100, output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)

        return x


class DomainClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 100)
        self.linear2 = nn.Linear(100, 2)  # 2 domains: source vs target
    
    def forward(self, x, alpha):
        x = grl(x, alpha)

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)

        return x
    

class DANN(nn.Module):
    def __init__(self, input_dim=3, output_dim=10):
        super().__init__()
        self.feature = FeatureExtractor(input_dim)
        self.label_predictor = LabelPredictor(48*4*4, output_dim)
        self.domain_classifier = DomainClassifier(48*4*4)
    
    def forward(self, x, alpha=1.0):
        features = self.feature(x)
        class_output = self.label_predictor(features)
        domain_output = self.domain_classifier(features, alpha)
        return class_output, domain_output