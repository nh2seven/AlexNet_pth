import os
import json
import yaml
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MiniImageNetDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None, split=None, conf_path="config.yaml"):
        with open(conf_path, "r") as f:  # Load config from root directory unless specified otherwise
            config = yaml.safe_load(f)

        with open(json_path, "r") as f:
            data = json.load(f)

        self.image_paths = data["image_names"]
        self.labels = data["image_labels"]
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # Meta-learning parameters
        self.n_way = config["meta"]["n_way"]
        self.k_shot = config["meta"]["k_shot"]
        self.q_queries = config["meta"]["q_queries"]

        # Group images by label
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.unique_labels) // self.n_way

    def load_image(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        selected_classes = random.sample(self.unique_labels, self.n_way)

        support_x = []
        support_y = []
        query_x = []
        query_y = []

        for class_idx, class_label in enumerate(selected_classes):
            class_indices = self.label_to_indices[class_label]
            selected_examples = random.sample(
                class_indices, self.k_shot + self.q_queries
            )
            support_indices = selected_examples[: self.k_shot]
            query_indices = selected_examples[self.k_shot :]

            # Load support images
            for idx in support_indices:
                support_x.append(self.load_image(idx))
                support_y.append(class_idx)

            # Load query images
            for idx in query_indices:
                query_x.append(self.load_image(idx))
                query_y.append(class_idx)

        support_x = torch.stack(support_x)
        support_y = torch.LongTensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.LongTensor(query_y)

        return support_x, support_y, query_x, query_y


def get_dataloaders(config, transform):
    # Create meta-datasets
    train_dataset = MiniImageNetDataset(
        json_path=config["data"]["json_base"],
        root_dir=config["data"]["root_dir"],
        transform=transform,
        split="train",
    )
    val_dataset = MiniImageNetDataset(
        json_path=config["data"]["json_val"],
        root_dir=config["data"]["root_dir"],
        transform=transform,
        split="val",
    )
    test_dataset = MiniImageNetDataset(
        json_path=config["data"]["json_test"],
        root_dir=config["data"]["root_dir"],
        transform=transform,
        split="test",
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    with open("config.yaml", "r") as f: # Specify config path file explicitly
        config = yaml.safe_load(f)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_loader, val_loader, test_loader = get_dataloaders(config, transform)

    for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(train_loader):
        print(f"\nEpisode {batch_idx + 1}:")
        print(f"Number of unique classes: {len(torch.unique(support_y))} (should be {config["meta"]["n_way"]})")

        print(f"Support set shape: {support_x.shape} (should be [1, {config["meta"]["n_way"]*config["meta"]["k_shot"]}, 3, 224, 224])")
        print(f"Support labels shape: {support_y.shape} (should be [1, {config["meta"]["n_way"]*config["meta"]["k_shot"]}])")
        print(f"Query set shape: {query_x.shape} (should be [1, {config["meta"]["n_way"]*config["meta"]["q_queries"]}, 3, 224, 224])")
        print(f"Query labels shape: {query_y.shape} (should be [1, {config["meta"]["n_way"]*config["meta"]["q_queries"]}])")

        if batch_idx >= 0:
            break

    print(f"\n[INFO] Train loader has {len(train_loader)} episodes")
    print(f"[INFO] Val loader has {len(val_loader)} episodes")
    print(f"[INFO] Test loader has {len(test_loader)} episodes")
