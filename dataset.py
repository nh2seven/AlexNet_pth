import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class MiniImageNetDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.image_paths = data["image_names"]
        self.labels = data["image_labels"]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


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

    # Datasets
    train_dataset = MiniImageNetDataset(json_path=JSON_BASE, root_dir=ROOT, transform=transform)
    val_dataset = MiniImageNetDataset(json_path=JSON_VAL, root_dir=ROOT, transform=transform)
    test_dataset = MiniImageNetDataset(json_path=JSON_TEST, root_dir=ROOT, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
