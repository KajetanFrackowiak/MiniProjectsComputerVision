import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset


def preprocess(example, train=True):
    """
    Preprocess a CIFAR-10 example.

    Args:
        example (dict): Dataset example with keys 'img' and 'label'.
                       Note: with_transform always provides batched data (lists).
        train (bool): Whether to apply random augmentation.

    Returns:
        dict: Preprocessed example with keys 'img' (float32, [-1, 1]) and 'label' (int32).
    """
    imgs = example["img"]
    labels = example["label"]

    # Process each image in the batch
    processed_images = []
    for img in imgs:
        # Convert PIL image to numpy array
        x = np.array(img, dtype=np.float32)
        processed_img = _preprocess_single_image(x, train)
        processed_images.append(processed_img)

    # Stack into batch
    x = np.stack(processed_images)
    y = np.array(labels)

    return {"img": x, "label": y}


def _preprocess_single_image(x, train=True):
    """
    Preprocess a single CIFAR-10 image.

    Args:
        x (np.ndarray): Image array (H, W, C).
        train (bool): Whether to apply random augmentation.

    Returns:
        np.ndarray: Preprocessed image (float32, [-1, 1]).
    """
    # Normalize to [0, 1]
    x = x / 255.0

    if train:
        # Pad 4 pixels on each side (32 -> 40)
        x = np.pad(x, ((4, 4), (4, 4), (0, 0)), mode="reflect")

        # Random crop back to 32x32
        h_start = np.random.randint(0, x.shape[0] - 32 + 1)
        w_start = np.random.randint(0, x.shape[1] - 32 + 1)
        x = x[h_start : h_start + 32, w_start : w_start + 32, :]

        # Random horizontal flip
        # x.shape is (H, W, C)
        if np.random.rand() > 0.5:
            x = x[:, ::-1, :]

        # Color jitter (brightness/contrast)
        x = x * np.random.uniform(0.9, 1.1)
        x = np.clip(x, 0, 1)

    # Normalize to [-1, 1]
    x = (x - 0.5) / 0.5

    return x


def load_cifar10(train=True):
    split = "train" if train else "test"
    ds = load_dataset("uoft-cs/cifar10", split=split)
    ds = ds.with_transform(lambda example: preprocess(example, train=train))
    return ds


def numpy_collate(batch):
    return {
        "img": np.stack([b["img"] for b in batch]),
        "label": np.stack([b["label"] for b in batch]),
    }


def dataloader(dataset, batch_size=64, shuffle=False):
    def _iterator():
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_indices = batch_indices.tolist()
            batch = [dataset[i] for i in batch_indices]
            yield numpy_collate(batch)

    return _iterator


def get_data(batch_size=64):
    train_ds = load_cifar10(train=True)
    test_ds = load_cifar10(train=False)
    train_loader = dataloader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = dataloader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


train_loader, _ = get_data()

for batch in train_loader():
    images = batch["img"]
    labels = batch["label"]
    break
