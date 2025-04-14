import yaml
import torch
from dataset import get_dataloaders
from alexnet import AlexNet, Checkpoint
from utils import Meta
from torch.optim import Adam
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

alex = AlexNet(classes=config["meta"]["n_way"])
meta = Meta(model=alex, device=device)
checkpoint = Checkpoint(device=device, model_dir="checkpoints")
optimizer = Adam(alex.parameters(), lr=0.001)

train_loader, val_loader, test_loader = get_dataloaders(config=config, transform=transform) 


# Entry point for the whole project
if __name__ == "__main__":
    choice = input("1: Train\n2: Resume\n3: Inference\nAnything Else: Exit\n\n-> ")

    if choice == "1":
        print("Starting training from scratch...")
        meta.train(dataloader=train_loader, optimizer=optimizer)
        checkpoint.save(model=alex, optimizer=optimizer, epoch=0)

    elif choice == "2":
        print("Resuming training from checkpoint...")
        epoch = checkpoint.load(model=alex, optimizer=optimizer)
        meta.train(dataloader=train_loader, optimizer=optimizer)
        checkpoint.save(model=alex, optimizer=optimizer, epoch=epoch)

    elif choice == "3":
        print("Starting inference...")
        checkpoint.load(model=alex)
        meta.evaluate(dataloader=test_loader)
    
    else:
        print("Exiting...\nRestart if you want to train or resume training.")
        exit(0)
