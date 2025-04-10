import os
import torch
import torch.nn.functional as F
from copy import deepcopy


class Checkpoint:
    def __init__(self, device, model_dir):
        self.device = device
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, name="checkpoint.pth"):
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, os.path.join(self.model_dir, name))

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
    def __init__(self, model, device, inner_lr=0.01, inner_steps=1, n_way=5, k_shot=1, q_queries=15):
        self.model = model
        self.device = device
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries

    def clone_model(self):
        return deepcopy(self.model)

    def forward_on_batch(self, model, x):
        return model(x)

    def compute_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def train(self, dataloader, optimizer, epochs):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(epochs):
            for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(dataloader):
                support_x, support_y = support_x.to(self.device), support_y.to(self.device)
                query_x, query_y = query_x.to(self.device), query_y.to(self.device)

                # Step 1: clone model
                adapted_model = self.clone_model()

                # Step 2: inner loop
                for _ in range(self.inner_steps):
                    logits = adapted_model(support_x)
                    loss = self.compute_loss(logits, support_y)
                    grads = torch.autograd.grad(loss, adapted_model.parameters(), create_graph=True)
                    for param, grad in zip(adapted_model.parameters(), grads):
                        param.data -= self.inner_lr * grad

                # Step 3: outer loop
                query_logits = adapted_model(query_x)
                query_loss = self.compute_loss(query_logits, query_y)
                optimizer.zero_grad()
                query_loss.backward()
                optimizer.step()

                acc1, acc5 = self.accuracy_topk(query_logits, query_y, topk=(1, 5))
                print(f"[Epoch {epoch} | Batch {batch_idx}] Loss: {query_loss.item():.4f} | Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}%")

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
        print(f"Validation Accuracy → Top-1: {top1_avg:.2f}% | Top-5: {top5_avg:.2f}%")
        return top1_avg, top5_avg

    def predict(self, image_tensor):
        self.model.eval()
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_labels = probs.topk(5)
            return top_labels[0].tolist(), top_probs[0].tolist()

    def accuracy_topk(self, logits, targets, topk=(1, 5)):
        """Computes the top-k accuracy"""
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()  # shape: [topk, batch]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            acc = (correct_k / batch_size) * 100.0
            res.append(acc.item())
        return res
