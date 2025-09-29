import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from datasets import load_dataset

ds = load_dataset("grodino/waterbirds")

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class DataWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]["image"]
        y = self.dataset[idx]["label"]
        g = self.dataset[idx]["place"]  # group

        if self.transform:
            x = self.transform(x)

        return x, y, g


def load_data(batch_size=64):
    train_data = DataWrapper(ds["train"], transform=train_transform)
    test_data = DataWrapper(ds["test"], transform=test_transform)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8
    )

    return train_loader, test_loader


class MixupStrategy:
    """Base class for different mixup strategies"""

    def __call__(self, x, y, g, alpha, **kwargs):
        raise NotImplementedError


class LISAMixup(MixupStrategy):
    """LISA: Learning Invariant Representations with mixup across domains/labels"""

    def __call__(self, x, y, g, alpha=0.4, psel=0.5):
        if alpha == 0:
            return x, y, y, 1.0  # No mixup

        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        batch_size = x.size(0)

        strategy = random.random() < psel

        mixed_indices = []
        for i in range(batch_size):
            if strategy:
                # intra-label, inter-domain: same label, different domain
                candidates = [
                    j for j in range(batch_size) if y[j] == y[i] and g[j] != g[i]
                ]
            else:
                # intra-domain, inter-label: different label, same domain
                candidates = [
                    j for j in range(batch_size) if y[j] != y[i] and g[j] == g[i]
                ]

            if candidates:
                j = random.choice(candidates)
            else:
                # fallback: random sample (but not self)
                j = random.choice([k for k in range(batch_size) if k != i])
            mixed_indices.append(j)

        mixed_indices = torch.tensor(mixed_indices)
        mixed_x = lam * x + (1 - lam) * x[mixed_indices]

        return mixed_x, y, y[mixed_indices], lam


class IntraLabelMixup(MixupStrategy):
    """Mixup only within same labels (safer for preserving semantics)"""

    def __call__(self, x, y, g, alpha=0.4, **kwargs):
        if alpha == 0:
            return x, y, y, 1.0

        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        batch_size = x.size(0)

        mixed_indices = []
        for i in range(batch_size):
            # Only mix with same label
            candidates = [j for j in range(batch_size) if y[j] == y[i] and j != i]
            if candidates:
                j = random.choice(candidates)
            else:
                j = i  # Keep original if no candidates
            mixed_indices.append(j)

        mixed_indices = torch.tensor(mixed_indices)
        mixed_x = lam * x + (1 - lam) * x[mixed_indices]

        return mixed_x, y, y[mixed_indices], lam


class IntraDomainMixup(MixupStrategy):
    """Mixup only within same domains"""

    def __call__(self, x, y, g, alpha=0.4, **kwargs):
        if alpha == 0:
            return x, y, y, 1.0

        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        batch_size = x.size(0)

        mixed_indices = []
        for i in range(batch_size):
            # Only mix with same domain
            candidates = [j for j in range(batch_size) if g[j] == g[i] and j != i]
            if candidates:
                j = random.choice(candidates)
            else:
                j = i  # Keep original if no candidates
            mixed_indices.append(j)

        mixed_indices = torch.tensor(mixed_indices)
        mixed_x = lam * x + (1 - lam) * x[mixed_indices]

        return mixed_x, y, y[mixed_indices], lam


class StandardMixup(MixupStrategy):
    """Standard random mixup"""

    def __call__(self, x, y, g, alpha=0.4, **kwargs):
        if alpha == 0:
            return x, y, y, 1.0

        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        batch_size = x.size(0)
        indices = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[indices]
        return mixed_x, y, y[indices], lam


class NoMixup(MixupStrategy):
    """No mixup - standard training"""

    def __call__(self, x, y, g, alpha=0.0, **kwargs):
        return x, y, y, 1.0


def get_mixup_strategy(strategy_name):
    strategies = {
        "lisa": LISAMixup(),
        "intra_label": IntraLabelMixup(),
        "intra_domain": IntraDomainMixup(),
        "standard": StandardMixup(),
        "none": NoMixup(),
    }
    return strategies.get(strategy_name.lower(), LISAMixup())


# Legacy function for backward compatibility
def mixup_data(x, y, g, alpha=0.4, psel=0.5):
    """Legacy LISA mixup function"""
    return LISAMixup()(x, y, g, alpha, psel)
