import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset


def _preprocess_single_image(image: jnp.ndarray, train: bool, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Preprocess a single image: normalize, augment (if train)
    Args:
        image: Input image as a numpy array of shape (H, W, C) with values in [0, 255]
        train: Whether to apply training augmentations
        key: JAX random key for augmentations
    Returns:
        Preprocessed image as a jnp.ndarray of shape (H, W, C) with values in [-1, 1]
    """
    x = image / 255.0  # normalize to [0,1]

    if train:
        key, k_crop, k_flip, k_bright, k_contrast = jax.random.split(key, 5)

        # Random crop
        pad = 4
        # [32, 32, 3] -> [40, 40, 3]
        x = jnp.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        h_start = jax.random.randint(k_crop, (), 0, x.shape[0] - 32 + 1)
        w_start = jax.random.randint(k_crop, (), 0, x.shape[1] - 32 + 1)
        # Crop back to [32, 32, 3]
        x = x[h_start:h_start+32, w_start:w_start+32, :]

        # Random horizontal flip
        x = jax.lax.cond(jax.random.uniform(k_flip) > 0.5,
                         lambda x: x[:, ::-1, :],
                         lambda x: x,
                         operand=x)

        # Random brightness
        # alpha > 1.0 makes image brighter, < 1.0 makes it darker
        alpha = jax.random.uniform(k_bright, minval=0.9, maxval=1.1)
        x = x * alpha

        # Random contrast
        # beta > 1.0 increases contrast, < 1.0 decreases contrast
        beta = jax.random.uniform(k_contrast, minval=0.9, maxval=1.1)
        mean = jnp.mean(x, axis=(0,1), keepdims=True)
        x = (x - mean) * beta + mean

        x = jnp.clip(x, 0.0, 1.0)

    # Normalize to [-1, 1]
    x = (x - 0.5) / 0.5
    return x


def load_cifar10(train: bool = True) -> dict:
    """Load CIFAR-10 dataset using Hugging Face datasets library
    Args:
        train: Whether to load the training split or test split
    """
    split = "train" if train else "test"
    dataset = load_dataset("cifar10", split=split)

    def convert_to_numpy(example):
        imgs = np.array(example["img"], dtype=np.float32)
        labels = np.array(example["label"])
        return {"img": imgs, "label": labels}

    dataset = dataset.map(convert_to_numpy, batched=True)
    return dataset

def preprocess_batch(images: jnp.ndarray, train: bool, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Preprocess a batch of images
    Args:
        images: Batch of images as a numpy array of shape (B, H, W, C) with values in [0, 255]
        train: Whether to apply training augmentations
        key: JAX random key for augmentations
    Returns:
        Preprocessed images as a jnp.ndarray of shape (B, H, W, C) with values in [-1, 1]
    """
    keys = jax.random.split(key, len(images))
    return jax.vmap(_preprocess_single_image, in_axes=(0,None,0))(images, train, keys)

def data_loader(dataset: dict, batch_size: int, train: bool = True, shuffle: bool = True, seed: int = 42) -> iter:
    """
    Create a data loader that yields batches of preprocessed images and labels
    Args:
        dataset: Dataset object from Hugging Face datasets library
        batch_size: Number of samples per batch
        train: Whether to apply training augmentations
        shuffle: Whether to shuffle the dataset each epoch
        seed: Random seed for shuffling and augmentations
    Returns:
        Iterator that yields tuples of (batch_images, batch_labels)
    """
    imgs = np.array(dataset["img"], dtype=np.float32)
    labels = np.array(dataset["label"])
    N = len(imgs)
    key = jax.random.PRNGKey(seed)
    
    def _iterator():
        indices = np.arange(N)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, N, batch_size):
            batch_idx = indices[start:start+batch_size]
            batch_imgs = imgs[batch_idx]
            batch_labels = labels[batch_idx]
            nonlocal key
            key, subkey = jax.random.split(key)
            if train:
                batch_imgs = preprocess_batch(batch_imgs, train, subkey)
            else:
                batch_imgs = (batch_imgs / 255.0 - 0.5)/0.5
            yield batch_imgs, batch_labels
    
    return _iterator