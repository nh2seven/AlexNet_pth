import yaml
import torch
from dataset import MiniImageNetDataset, get_dataloaders
from alexnet import AlexNet, Checkpoint
from utils import Meta
from torch.utils.data import DataLoader
from torchvision import transforms

# Load config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define the transform for the dataset
transform = transforms.Compose(
    [
        transforms.Resize(config["data"]["image_size"]),
        transforms.CenterCrop(config["data"]["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = AlexNet(classes=100)
meta_learner = Meta(model=model, device=device)


# Entry point for the whole project
if __name__ == "__main__":
    pass
    # optimizer = torch.optim.Adam(model.parameters(), lr=config["meta"]["outer_lr"])
    # checkpoint = Checkpoint(device=device, model_dir=config["training"]["checkpoint_dir"])
    # meta_learner.train(train_loader, optimizer)
    # checkpoint.save(model, optimizer, config["meta"]["epochs"], name="final.pth")
