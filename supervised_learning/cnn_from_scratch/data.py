from datasets import load_dataset

def load_data():
    dataset = load_dataset("ylecun/mnist", split="train")
    return dataset.shuffle()

def get_data():
    dataset = load_data()
    images = dataset["image"]
    labels = dataset["label"]
    return images, labels

