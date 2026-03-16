import cv2
import grain.python as grain
import numpy as np
from datasets import load_dataset


def cpu_preprocess(features: dict) -> dict:
    img = np.array(features["image"])
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    if img.shape[:2] != (32, 32):
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)

    raw_label = features["label"]
    if isinstance(raw_label, dict):
        label = raw_label["digit"][0]
    else:
        label = raw_label

    return {"image": img.astype(np.float32) / 255.0, "label": int(label)}


def get_data_loader(
    batch_size: int, seed1: int, seed2: int, split: str = "train"
) -> grain.IterDataset:
    source1 = load_dataset("ylecun/mnist", split=split)
    source2 = load_dataset("Genius-Society/svhn", split=split)

    ds1 = (
        grain.MapDataset.source(source1)
        .filter(lambda x: int(x["label"]) <= 4)
        .seed(seed1)
        .shuffle()
        .repeat()
    )

    ds2 = (
        grain.MapDataset.source(source2)
        .filter(lambda x: int(x["label"]["digit"][0]) > 4)
        .seed(seed2)
        .shuffle()
        .repeat()
    )

    ds = grain.MapDataset.mix([ds1, ds2], weights=[0.7, 0.3])

    ds = ds.to_iter_dataset()
    ds = ds.map(cpu_preprocess)
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds
