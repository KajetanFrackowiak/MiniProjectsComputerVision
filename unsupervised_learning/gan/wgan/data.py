import tensorflow as tf
from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")

class DataWrapper:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset[idx]["img"]
        y = self.dataset[idx]["label"]

        if self.transform:
            x, y = self.transform({"img": x, "label": y})
        
        return x, y

    def convert_to_dataset(self, batch_size=64, shuffle=False):
        dataset = tf.data.Dataset.from_generator(
            lambda: (self[i] for i in range(len(self))),
            output_signature=(
                tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64)
            )
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        

def transform(example, training=True):
    image = tf.cast(example["img"], tf.float32) / 255.0 # ToTensor
    
    if training:
        # Pad 4 pixels and random crop
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, size=[32, 32, 3])

        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Color jitter
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Normalize to [-1, 1]
    image = (image - 0.5) / 0.5
    
    label = tf.cast(example["label"], tf.int64)
    
    return image, label


def load_data(batch_size):
    train_wrapper = DataWrapper(ds["train"], transform=lambda example: transform(example, training=True))
    test_wrapper = DataWrapper(ds["test"], transform=lambda example: transform(example, training=False))

    train_loader = train_wrapper.convert_to_dataset(batch_size=batch_size, shuffle=True)
    test_loader = test_wrapper.convert_to_dataset(batch_size=batch_size, shuffle=False)

    return train_loader, test_loader