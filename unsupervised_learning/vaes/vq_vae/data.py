from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]["img"]
        if self.transform:
            x = self.transform(x)
        
        return x
    

def load_data(batch_size=64):
    train_data = DatasetWrapper(ds["train"], transform=transform)
    test_data = DatasetWrapper(ds["test"], transform=transform)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader

