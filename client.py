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

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # We create a fresh model so Opacus doesn't double-wrap it across rounds
        local_model = Net().to(self.device)
        params_dict = zip(local_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        local_model.load_state_dict(state_dict, strict=True)
        
        optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
        
        # Train with DP and get the epsilon budget
        epsilon = train(local_model, self.trainloader, optimizer, epochs=5, device=self.device, mu=0.01)
        print(f"🔒 [Hospital Client] Privacy Budget Used: ε = {epsilon:.2f}")
        
        # Safely unwrap the Opacus model to get the parameters back
        wrapped_state_dict = local_model._module.state_dict() if hasattr(local_model, '_module') else local_model.state_dict()
        new_parameters = [val.cpu().numpy() for _, val in wrapped_state_dict.items()]
        
        return new_parameters, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, f1 = test(self.model, self.valloader, device=self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy), "f1_score": float(f1)}
