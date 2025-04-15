import os
import torch


# Checkpoint class to handle saving and loading model checkpoints
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("epoch", 0)