from typing import Any, Protocol

import cv2
import grain.python as grain
import numpy as np
from datasets import load_dataset


class PreprocessFn(Protocol):
    def __call__(self, sample: dict[str, Any], *, target_size: int) -> dict[str, Any]: ...


def cifar10_preprocess(sample, target_size: int = 32):
    # Unikamy zbędnego kopiowania i resize, jeśli rozmiar się zgadza
    image = np.array(sample["img"])
    if image.shape[0] != target_size or image.shape[1] != target_size:
        image = cv2.resize(image, (target_size, target_size))

    # Przeskalowanie do [-1, 1]
    image = image.astype(np.float32) / 127.5 - 1.0
    return {"image": image}


def celeba_preprocess(sample, target_size=64):
    image = np.array(sample["image"])

    # Efektywny Center Crop
    h, w = image.shape[:2]
    min_size = min(h, w)
    start_h = (h - min_size) // 2
    start_w = (w - min_size) // 2
    image = image[start_h : start_h + min_size, start_w : start_w + min_size]

    # INTER_AREA jest najlepsze do zmniejszania obrazów (zapobiega aliasingowi)
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    image = (image.astype(np.float32) / 127.5) - 1.0
    return {"image": image}


def get_data_loader(
    dataset_path: str,
    batch_size: int,
    seed: int,
    split: str,
    preprocess_fn: PreprocessFn,
    target_size: int,
    repeat: bool = True,
):
    source = load_dataset(dataset_path, split=split)

    ds = grain.MapDataset.source(source)

    ds = ds.shuffle(seed=seed)

    if repeat:
        ds = ds.repeat()

    ds = ds.map(lambda x: preprocess_fn(x, target_size=target_size))

    it_ds = ds.to_iter_dataset()
    it_ds = it_ds.batch(batch_size, drop_remainder=(split == "train"))

    return it_ds
