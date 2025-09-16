import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

ds = load_dataset("ylecun/mnist")


def normalize_to_0_1(x):
    x = x.float()  # ensure float
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:  # avoid division by zero
        x = (x - x_min) / (x_max - x_min)
    else:
        x = torch.zeros_like(x)
    return x


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: normalize_to_0_1(x)), # [0,255] -> [0, 1]
        transforms.Lambda(lambda x: x.view(-1)),
    ]
)


def transform_batch(batch):
    return {"pixel_values": [transform(img) for img in batch["image"]]}


def load_data():
    train_ds = ds["train"].map(transform_batch, batched=True, load_from_cache_file=False)
    train_ds = train_ds.remove_columns(["image", "label"])
    test_ds = ds["test"].map(transform_batch, batched=True, load_from_cache_file=False)
    test_ds = test_ds.remove_columns(["image", "label"])

    train_ds.set_format(type="torch", columns=["pixel_values"])
    test_ds.set_format(type="torch", columns=["pixel_values"])

    trainloader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8)
    testloader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8)

    return trainloader, testloader
