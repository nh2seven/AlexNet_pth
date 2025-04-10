import yaml
import torch
from dataset import MiniImageNetDataset
from alexnet import AlexNet
from utils import Meta, Checkpoint
from torch.utils.data import DataLoader
from torchvision import transforms

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(classes=100)

transform = transforms.Compose(
    [
        transforms.Resize(config["data"]["image_size"]),
        transforms.CenterCrop(config["data"]["image_size"]),
        transforms.ToTensor(),
    ]
)

train_dataset = MiniImageNetDataset(
    json_path=config["data"]["json_base"],
    root_dir=config["data"]["root_dir"],
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

meta_learner = Meta(model=model, device=device)

if __name__ == "__main__":
    optimizer = torch.optim.Adam(model.parameters(), lr=config["meta"]["outer_lr"])
    checkpoint = Checkpoint(device=device, model_dir=config["training"]["checkpoint_dir"])
    meta_learner.train(train_loader, optimizer)
    checkpoint.save(model, optimizer, config["meta"]["epochs"], name="final.pth")
