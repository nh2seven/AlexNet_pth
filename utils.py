import torch
import torch.nn.functional as F
from copy import deepcopy
import yaml

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Class to handle the meta-learning process
class Meta:
    def __init__(self, model, device, conf_path="config.yaml"):
        with open(conf_path, "r") as f: # Load config from root directory unless specified otherwise
            config = yaml.safe_load(f)

        self.model = model
        self.device = device
        self.epochs = config["meta"]["epochs"]
        self.inner_lr = config["meta"]["inner_lr"]
        self.inner_steps = config["meta"]["inner_steps"]
        self.n_way = config["meta"]["n_way"]
        self.k_shot = config["meta"]["k_shot"]
        self.q_queries = config["meta"]["q_queries"]

    # Function to copy the model for each task/episode
    def clone_model(self):
        return deepcopy(self.model)

    # Function to compute the loss using cross-entropy
    def compute_loss(self, logits, labels):
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        return F.cross_entropy(logits, labels)

    # Function to train the model using meta-learning; each episode consists of a support set and a query set
    def train(self, dataloader, optimizer):
        self.model.to(self.device)
        self.model.train()
        total_loss = 0

        for batch_idx, (support_x, support_y, query_x, query_y) in enumerate(dataloader):
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            support_logits = self.model(support_x)
            query_logits = self.model(query_x)

            # Compute loss
            support_loss = self.compute_loss(support_logits, support_y)
            query_loss = self.compute_loss(query_logits, query_y)
            loss = support_loss + query_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        return total_loss / len(dataloader)

    # Function to evaluate the model on the val set
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for support_x, support_y, query_x, query_y in dataloader:
                support_x = support_x.to(self.device)
                support_y = support_y.to(self.device)
                query_x = query_x.to(self.device)
                query_y = query_y.to(self.device)

                # Forward pass
                query_logits = self.model(query_x)
                
                # Compute accuracy
                pred = query_logits.view(-1, query_logits.size(-1)).argmax(dim=1)
                target = query_y.view(-1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        accuracy = 100. * correct / total
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        return accuracy

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


# Invalid entry point
if __name__ == "__main__":
    exit(0)
