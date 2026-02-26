import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def partition_data(num_clients, alpha=0.5):
    """
    Partitions data using Dirichlet distribution to simulate 
    real-world medical data heterogeneity (Non-IID).
    """
    dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    
    classes = len(dataset.classes)
    label_indices = {i: np.where(dataset.targets == i)[0] for i in range(classes)}
    
    client_data_indices = [[] for _ in range(num_clients)]
    
    for k in range(classes):
        # Dirichlet distribution to skew the labels
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Split indices of class k among clients
        split_indices = np.split(label_indices[k], 
                                 (np.cumsum(proportions) * len(label_indices[k])).astype(int)[:-1])
        
        for i in range(num_clients):
            client_data_indices[i].extend(split_indices[i])
            
    return [Subset(dataset, indices) for indices in client_data_indices]

from opacus import PrivacyEngine

def train(model, train_loader, optimizer, epochs, mu=0.01):
    """
    mu: The proximal term constant for FedProx.
    It penalizes local updates that stray too far from the Global Model.
    """
    global_weight = [param.data.clone().detach() for param in model.parameters()]
    model.train()
    
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            
            # Standard CrossEntropy Loss
            loss = torch.nn.functional.nll_loss(output, target)
            
            # FEDPROX TERM: ||w - w_t||^2
            proximal_term = 0.0
            for param, g_param in zip(model.parameters(), global_weight):
                proximal_term += (param - g_param).norm(2)
            
            loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim

    # Define a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(-1, 28*28)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return torch.log_softmax(x, dim=1)

    print("Partitioning data...")
    try:
        datasets_list = partition_data(num_clients=5, alpha=0.5)
        print(f"Successfully partitioned data into {len(datasets_list)} subsets.")
    except Exception as e:
        print(f"Error during partition_data: {e}")
        exit(1)

    print("Setting up training...")
    try:
        dataset = datasets_list[0]
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        print("Starting training...")
        train(model, train_loader, optimizer, epochs=1)
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        exit(1)
