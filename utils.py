import os
import torch
import torch.nn.functional as F
from copy import deepcopy
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class Checkpoint:
    def __init__(self, device, model_dir):
        self.device = device
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    # Function to save the model and optimizer state to a checkpoint
    def save(self, model, optimizer, epoch, name="checkpoint.pth"):
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, os.path.join(self.model_dir, name))

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


class Meta:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.epochs = config["meta"]["epochs"]
        self.inner_lr = config["meta"]["inner_lr"]
        self.inner_steps = config["meta"]["inner_steps"]
        self.n_way = config["meta"]["n_way"]
        self.k_shot = config["meta"]["k_shot"]
        self.q_queries = config["meta"]["q_queries"]

    # Function to compute the number of parameters in the model
    def clone_model(self):
        return deepcopy(self.model)

    # Function to perform a forward pass on a batch of data
    def forward_on_batch(self, model, x):
        return model(x)

    # Function to compute the loss using cross-entropy
    def compute_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    # Function to train the model using meta-learning; each episode consists of a support set and a query set
    def train(self, dataloader, optimizer):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(dataloader):
                support_x, support_y = support_x.to(self.device), support_y.to(self.device)
                query_x, query_y = query_x.to(self.device), query_y.to(self.device)

                adapted_model = self.clone_model()

                for _ in range(self.inner_steps):
                    logits = adapted_model(support_x)
                    loss = self.compute_loss(logits, support_y)
                    grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)
                    for param, grad in zip(adapted_model.parameters(), grads):
                        param.data -= self.inner_lr * grad

                query_logits = adapted_model(query_x)
                query_loss = self.compute_loss(query_logits, query_y)
                optimizer.zero_grad()
                query_loss.backward()
                optimizer.step()

                acc1, acc5 = self.accuracy_topk(query_logits, query_y, topk=(1, 5))
                print(f"[Epoch {epoch} | Batch {batch_idx}] Loss: {query_loss.item():.4f} | Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}%")

    # Function to evaluate the model on the val set
    def evaluate(self, dataloader):
        self.model.eval()
        top1_total, top5_total, total = 0, 0, 0
        with torch.no_grad():
            for query_x, query_y in dataloader:
                query_x, query_y = query_x.to(self.device), query_y.to(self.device)
                logits = self.model(query_x)
                acc1, acc5 = self.accuracy_topk(logits, query_y, topk=(1, 5))
                top1_total += acc1 * query_x.size(0)
                top5_total += acc5 * query_x.size(0)
                total += query_x.size(0)

        top1_avg = top1_total / total
        top5_avg = top5_total / total
        print(f"Validation Accuracy -> Top-1: {top1_avg:.2f}% | Top-5: {top5_avg:.2f}%")
        return top1_avg, top5_avg

    # Function to predict the class of a single image
    def predict(self, image_tensor):
        self.model.eval()
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_labels = probs.topk(5)
            return top_labels[0].tolist(), top_probs[0].tolist()

    # Function to compute the top-k accuracy
    def accuracy_topk(self, logits, targets, topk=(1, 5)):
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            acc = (correct_k / batch_size) * 100.0
            res.append(acc.item())
        return res


if __name__ == "__main__":
    print("heh")
    exit(0)