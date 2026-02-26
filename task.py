import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """A robust CNN for medical classification."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, trainloader, optimizer, epochs, device, mu):
    """Train with FedProx regularization."""
    model.train()
    global_params = [p.data.clone() for p in model.parameters()]
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            # FedProx term
            proximal_term = 0.0
            for p, g_p in zip(model.parameters(), global_params):
                proximal_term += (p - g_p).norm(2)
            loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()

def test(model, testloader, device):
    """Validate model performance."""
    model.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += F.nll_loss(outputs, labels, reduction="sum").item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    return loss / len(testloader.dataset), correct / len(testloader.dataset)