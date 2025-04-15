import torch
from torchvision import transforms
from PIL import Image
from models.alexnet import AlexNet


def load_model(checkpoint_path):
    model = AlexNet(classes=100)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
    model.eval()
    return model


def predict(model, image_path):
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(image)
    probs = torch.softmax(logits, dim=1)
    top5 = torch.topk(probs, k=5)
    return top5
