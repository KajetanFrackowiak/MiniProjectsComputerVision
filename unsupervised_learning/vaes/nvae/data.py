import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")

transform = transforms.Compose([
    transforms.ToTensor(),  # [0,255] -> [0,1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # [0,1] -> [-1,1]
])

# Apply transform in column pixel_values
def transform_batch(batch):
    return {"pixel_values": [transform(img) for img in batch["img"]]}

def load_data():
    train_ds = ds["train"].map(transform_batch, batched=True, batch_size=2000)
    test_ds = ds["test"].map(transform_batch, batched=True, batch_size=2000)

    train_ds = train_ds.remove_columns(["img", "label"])
    test_ds = test_ds.remove_columns(["img", "label"])

    train_ds.set_format(type="torch", colums=["pixel_values"])
    test_ds.set_format(type="torch", columns=["pixel_values"])

    trainloader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=8
    )
    testloader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=8
    )
    
    return trainloader, testloader

if __name__ == "__main__":
    trainloader, testloader = load_data()