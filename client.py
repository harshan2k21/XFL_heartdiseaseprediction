import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from task import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader):
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # mu=0.01 for FedProx
        train(self.model, self.trainloader, optimizer, epochs=5, device=self.device, mu=0.01)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, device=self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}
