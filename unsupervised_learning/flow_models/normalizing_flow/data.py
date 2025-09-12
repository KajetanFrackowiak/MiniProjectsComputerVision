from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # [0,1] -> [-1,1]
])

def transform_batch(batch):
    return {"pixel_values": [transform(img) for img in batch["img"]]}

def load_data(batch_size=64):

    train_ds = ds["train"].map(transform_batch, batched=True)
    train_ds = train_ds.remove_columns(["img", "label"])
    test_ds = ds["test"].map(transform_batch, batched=True)
    test_ds = test_ds.remove_columns(["img", "label"])

    train_ds.set_format(type="torch", columns=["pixel_values"])
    test_ds.set_format(type="torch", columns=["pixel_values"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=16)

    return train_loader, test_loader


    
