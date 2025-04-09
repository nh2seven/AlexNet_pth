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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(classes=100)
    summary(model, input_size=(1, 3, 224, 224), device=device)
