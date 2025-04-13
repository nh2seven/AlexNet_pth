import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


# Custom Dataset for MiniImageNet
class MiniImageNetDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None, split=None):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.image_paths = data["image_names"]
        self.labels = data["image_labels"]
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Function to get DataLoaders according to filelists in data/filelists
def get_dataloaders(config, transform):
    # Create datasets
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
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Paths
    ROOT = "data"
    JSON_BASE = "data/filelists/base.json"
    JSON_VAL = "data/filelists/val.json"
    JSON_TEST = "data/filelists/novel.json"

    # Train dataset and loader
    train_dataset = MiniImageNetDataset(
        json_path=JSON_BASE,
        root_dir=ROOT,
        transform=transform,
        split="train",
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(f"[INFO] Loaded {len(train_dataset)} images for split: {train_dataset.split}")

    # Validation dataset and loader
    val_dataset = MiniImageNetDataset(
        json_path=JSON_VAL,
        root_dir=ROOT,
        transform=transform,
        split="val",
    )
    val_loader = DataLoader(val_dataset, batch_size=64)
    print(f"[INFO] Loaded {len(val_dataset)} images for split: {val_dataset.split}")

    # Test dataset and loader
    test_dataset = MiniImageNetDataset(
        json_path=JSON_TEST,
        root_dir=ROOT,
        transform=transform,
        split="test",
    )
    test_loader = DataLoader(test_dataset, batch_size=64)
    print(f"[INFO] Loaded {len(test_dataset)} images for split: {test_dataset.split}")
