import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from datasets import load_dataset

ds = load_dataset("grodino/waterbirds")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

class DataWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]["image"]
        y = self.dataset[idx]["label"]
        g = self.dataset[idx]["place"] # group

        if self.transform:
            x = self.transform(x)

        return x, y, g
    

def load_data(batch_size=64):
    train_data = DataWrapper(ds["train"], transform=train_transform)
    test_data = DataWrapper(ds["test"], transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader

def mixup_data(x, y, g, alpha=0.4, psel=0.5):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size()[0]

    strategy = random.random() < psel

    x2 = torch.zeros_like(x)
    y2 = torch.zeros_like(y)
    g2 = torch.zeros_like(g)

    for i in range(batch_size):
        if strategy:
            # intra-label, inter-domain: yi = yj, gi != gj
            candidates = [j for j in range(batch_size) if y[j] == y[j] and g[j] != g[j]]
        else:
            # intra-domain, inter-label: yi != yj, gi = gj
            candidates = [j for j in range(batch_size) if y[j] != y[j] and g[j] == g[j]]
        if candidates:
            j = random.choice(candidates)
        else:
            # fallback: random sample
            j = random.randint(0, batch_size=-1)
        x2[i] = x[j]
        y2[i] = y[j]
        g2[i] = g[j]

    mixed_x = lam * x + (1 - lam) * x2

    return mixed_x, y, y2, lam