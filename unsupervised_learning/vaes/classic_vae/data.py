import torch
import torchvision.transforms as transforms

from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # Normalize RGB
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def transform_batch(batch):
    # transforms.ToTensor() returns a torch.Tensor which isn't directly
    # JSON-serializable for the datasets arrow table. Convert to numpy so
    # the column can be stored. We'll call set_format(type='torch') below
    # so DataLoader yields torch.Tensors again.
    return {"pixel_values": [transform(img).numpy() for img in batch["img"]]}


def load_data():
    train_ds = ds["train"].map(transform_batch, batched=True)
    train_ds = train_ds.remove_columns(["img", "label"])
    test_ds = ds["test"].map(transform_batch, batched=True)
    test_ds = test_ds.remove_columns(["img", "label"])
    print(train_ds)

    # Make the dataset return torch tensors for the 'pixel_values' column
    train_ds.set_format(type="torch", columns=["pixel_values"])
    test_ds.set_format(type="torch", columns=["pixel_values"])

    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=8
    )

    testloader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=8
    )

    return trainloader, testloader
