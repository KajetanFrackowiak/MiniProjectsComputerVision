import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import random
from collections import defaultdict

ds = load_dataset("GATE-engine/omniglot")


transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
    

class FewShotDataset(Dataset):
    def __init__(self, dataset, n_way, k_shot, q_query, transform=None):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.transform = transform

        # group indices by class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.dataset["label"]):
            self.class_to_indices[label].append(idx)

    def __len__(self):
        # arbitrary for large number of episodes
        return 100000

    def __getitem__(self, _):
        # sample N classes
        classes = random.sample(list(self.class_to_indices.keys()), self.n_way)

        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for i, cls in enumerate(classes):
            indices = random.sample(self.class_to_indices[cls], self.k_shot + self.q_query)
            support_idx = indices[:self.k_shot]
            query_idx = indices[self.k_shot:]

            for idx in support_idx:
                img = self.dataset[idx]["image"]
                if self.transform:
                    img = self.transform(img)
                support_images.append(img)
                support_labels.append(i)
            
            for idx in query_idx:
                img = self.dataset[idx]["image"]
                if self.transform:
                    img = self.transform(img)
                query_images.append(img)
                query_labels.append(i)

        return {
            "support_images": torch.stack(support_images),
            "support_labels": torch.tensor(support_labels),
            "query_images": torch.stack(query_images),
            "query_labels": torch.tensor(query_labels),
        }

def transform_batch(batch):
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch

def load_data(n_way, k_shot, q_query):
    train_labels = set(range(964))
    test_labels = set(range(964, 1623))
    
    # split dataset
    train_ds = ds["full"].filter(lambda example: example["label"] in train_labels, load_from_cache_file=False)
    test_ds = ds["full"].filter(lambda example: example["label"] in test_labels, load_from_cache_file=False)
    
    fewshot_train = FewShotDataset(train_ds, n_way=n_way, k_shot=k_shot, q_query=q_query, transform=transform)
    fewshot_test = FewShotDataset(test_ds, n_way=n_way, k_shot=k_shot, q_query=q_query, transform=transform)
    
    train_loader = DataLoader(fewshot_train, batch_size=1, shuffle=True)
    test_loader = DataLoader(fewshot_test, batch_size=1, shuffle=False)

    return train_loader, test_loader

