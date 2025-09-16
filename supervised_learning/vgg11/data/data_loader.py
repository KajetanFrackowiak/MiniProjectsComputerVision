import torch
import torchvision
import yaml
from torchvision import transforms


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_data_loaders(batch_size=64, num_workers=4):
    config = load_config()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=config["data"]["cifar10"], train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=config["data"]["cifar10"], train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader
