import torch
import torch.nn as nn
from torchinfo import summary


# AlexNet with modified final layer
class AlexNet(nn.Module):
    def __init__(self, classes=100):
        super(AlexNet, self).__init__()

        # Input size: 224x224
        self.features = nn.Sequential(
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

        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Classifier
        self.classifier = nn.Sequential(
            # Block 6
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # Block 7
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Block 8
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 5:
            tasks, shots, c, h, w = x.shape
            x = x.view(-1, c, h, w)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if len(original_shape) == 5:
            x = x.view(tasks, shots, -1)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(classes=100).to(device)
    
    print("\nTesting standard input (batch, channels, height, width):")
    summary(model, input_size=(1, 3, 224, 224), device=device)
    
    print("\nTesting meta-learning input (tasks, shots, channels, height, width):")
    x = torch.randn(1, 5, 3, 224, 224).to(device)
    output = model(x)
    print(f"Meta-learning output shape: {output.shape}")