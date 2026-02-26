import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

class Net(nn.Module):
    def __init__(self, input_dim=13): # Default to 13 for Heart Disease
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train(model, trainloader, optimizer, epochs, device, mu):
    model.train()
    criterion = nn.CrossEntropyLoss()
    global_params = [p.data.clone() for p in model.parameters()]
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            
            proximal_term = 0.0
            for p, g_p in zip(model.parameters(), global_params):
                proximal_term += (p - g_p).norm(2)
            loss += (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()

def test(model, testloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Clinical Metrics
    acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return loss / len(testloader), acc, f1
