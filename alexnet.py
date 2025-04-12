import os
import torch
import torch.nn as nn
from torchinfo import summary


# AlexNet with modified final layer
class AlexNet(nn.Module):
    def __init__(self, classes=100):
        super(AlexNet, self).__init__()

        # Input size: 224x224
        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Block 5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Infer feature shape
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            n_features = self.model(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            # Block 6
            nn.Dropout(),
            nn.Linear(n_features, 4096),
            nn.ReLU(inplace=True),
            # Block 7
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Block 8
            nn.Linear(4096, classes),
        )
        # Output size: classes

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Checkpoint class to save and load model states
class Checkpoint:
    def __init__(self, device, model_dir):
        self.device = device
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    # Function to save the model and optimizer state to a checkpoint
    def save(self, model, optimizer, epoch, name="checkpoint.pth"):
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(self.model_dir, name),
        )

    # Function to load the model and optimizer state from a checkpoint
    def load(self, model, optimizer=None, name="checkpoint.pth"):
        path = os.path.join(self.model_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("epoch", 0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(classes=100)
    summary(model, input_size=(1, 3, 224, 224), device=device)
