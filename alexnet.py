import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models


# AlexNet with modified final layer
class AlexNet(nn.Module):
    def __init__(self, classes=100, pre=True, freeze=False):
        super(AlexNet, self).__init__()

        # Block 1-5
        alex_pt = models.alexnet(pretrained=pre)
        self.features = alex_pt.features

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

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

        nn.init.xavier_uniform_(self.classifier[6].weight)
        nn.init.constant_(self.classifier[6].bias, 0)

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


# Get model summary
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(classes=100, pre=True, freeze=False).to(device)

    print("\nTesting standard input (batch, channels, height, width):")
    summary(model, input_size=(1, 3, 224, 224), device=device)

    print("\nTesting meta-learning input (tasks, shots, channels, height, width):")
    x = torch.randn(1, 5, 3, 224, 224).to(device)
    output = model(x)
    print(f"Meta-learning output shape: {output.shape}")
