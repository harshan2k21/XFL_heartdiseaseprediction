import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from opacus import PrivacyEngine
import warnings

# Suppress opacus warnings for clean logs
warnings.filterwarnings("ignore")

class Net(nn.Module):
    def __init__(self, input_dim=13): 
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
    
    # --- NEW: Inject Differential Privacy ---
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=0.1, # The amount of "fog" we add to protect identity
        max_grad_norm=1.0,    # Caps the max influence of one patient
    )
    
    # Clone parameters for FedProx
    global_params = [p.data.clone() for p in model.parameters()]
    
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            
            # FedProx term
            proximal_term = 0.0
            for p, g_p in zip(model.parameters(), global_params):
                proximal_term += (p - g_p).norm(2)
            loss += (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            
    # Calculate Privacy Budget used (Epsilon)
    # Delta is typically 1 / size of dataset
    epsilon = privacy_engine.get_epsilon(delta=1e-3)
    return epsilon

def test(model, testloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return loss / len(testloader), acc, f1
