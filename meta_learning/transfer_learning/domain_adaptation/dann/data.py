from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

ds_mnist = load_dataset("ylecun/mnist")
ds_mnist_m = load_dataset("Mike0307/MNIST-M")

transform_mnist = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1x28x28 -> 3x28x28
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

transform_mnist_m = transforms.Compose(
    [
        transforms.Resize((28, 28)), # 32x32 -> 28x28
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]["image"]
        y = self.dataset[idx]["label"]
        if self.transform:
            x = self.transform(x)

        return x, y


def load_data(batch_size):
    train_source = DatasetWrapper(ds_mnist["train"], transform=transform_mnist)
    test_source = DatasetWrapper(ds_mnist["test"], transform=transform_mnist)
    train_target = DatasetWrapper(ds_mnist_m["train"], transform=transform_mnist_m)
    test_target = DatasetWrapper(ds_mnist_m["test"], transform=transform_mnist_m)

    train_source_loader = DataLoader(
        train_source, batch_size=batch_size, shuffle=True, num_workers=16
    )
    test_source_loader = DataLoader(
        test_source, batch_size=batch_size, shuffle=False, num_workers=16
    )
    train_target_loader = DataLoader(
        train_target, batch_size=batch_size, shuffle=True, num_workers=16
    )
    test_target_loader = DataLoader(
        test_target, batch_size=batch_size, shuffle=False, num_workers=16
    )

    return (
        train_source_loader,
        test_source_loader,
        train_target_loader,
        test_target_loader,
    )
