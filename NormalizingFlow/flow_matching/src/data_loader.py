from typing import Any, Protocol, Iterator

import grain.python as grain
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset


class PreprocessFn(Protocol):
    def __call__(self, sample: dict[str, Any], target_size: int) -> dict[str, Any]: ...


def cifar10_preprocess(sample: dict[str, Any], target_size: int = 32) -> dict[str, Any]:
    image = np.array(sample["img"])  #  PIL -> numpy array for cv2 processing
    if image.shape[0] != target_size or image.shape[1] != target_size:
        image = cv2.resize(image, (target_size, target_size))
    image = image.astype(np.float32) / 127.5 - 1.0  #  [0, 255] -> [-1, 1]
    return {"image": image}


def celeba_preprocess(sample: dict[str, Any], target_size: int = 64) -> dict[str, Any]:
    image = np.array(sample["image"])  #  PIL -> numpy array for cv2 processing

    # image.shape: [h, w, c]
    h, w = image.shape[:2]  # [178, 218, 3]
    min_size = min(h, w)  # 178
    # Center crop
    start_h = (h - min_size) // 2  # 0
    start_w = (w - min_size) // 2  # 20
    image = image[
        start_h : start_h + min_size, start_w : start_w + min_size
    ]  # [0:178, 20:198, 3]

    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 127.5 - 1.0  #  [0, 255] -> [-1, 1]
    return {"image": image}


def get_data_loader(
    dataset_path: str,
    batch_size: int,
    seed: int,
    split: str,
    preprocess_fn: PreprocessFn,
    target_size: int,
    repeat: bool = True,
) -> Iterator[dict[str, Any]]:
    source = load_dataset(dataset_path, split=split)

    ds = grain.MapDataset.source(source).shuffle(seed=seed)
    if repeat:
        ds = ds.repeat()
    ds = ds.map(lambda x: preprocess_fn(x, target_size=target_size))

    it_ds = ds.to_iter_dataset()
    it_ds = it_ds.batch(batch_size, drop_remainder=(split == "train"))
    return iter(it_ds)


if __name__ == "__main__":
    it_ds = get_data_loader(
        dataset_path="uoft-cs/cifar10",
        batch_size=16,
        seed=0,
        split="train",
        preprocess_fn=cifar10_preprocess,
        target_size=32,
        repeat=False,
    )

    def plot(batch):
        images = batch["image"]  # [16, 32, 32, 3]
        images = (images + 1.0) * 127.5  # [-1, 1] -> [0, 255]
        images = images.astype(np.uint8)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            ax = axes[i // 4, i % 4]
            ax.imshow(images[i])
            ax.axis("off")
        plt.savefig("cifar10_samples.png")
        plt.close()

    for batch in it_ds:
        print(batch["image"].shape)  # [16, 32, 32, 3]
        plot(batch)
        break
