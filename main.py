import flwr as fl
import torch
from collections import OrderedDict
from task import Net
from client import FlowerClient
from data_partining import partition_data
from torch.utils.data import DataLoader

def weighted_average(metrics):
    # Calculate weighted accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    # NEW: Calculate weighted F1-Score
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    
    examples = [num_examples for num_examples, _ in metrics]
    
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "f1_score": sum(f1_scores) / sum(examples)
    }

def get_evaluate_fn():
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model = Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        torch.save(model.state_dict(), "global_model.pth")
        print(f"\n💾 [Server] Saved Global Model to 'global_model.pth' at Round {server_round}")
        return None 
    return evaluate

print("🚀 Initializing Federated Heart Disease Prediction (Research Mode)...")

NUM_CLIENTS = 5
trainsets = partition_data(num_clients=NUM_CLIENTS)

def client_fn(cid: str):
    trainloader = DataLoader(trainsets[int(cid)], batch_size=16, shuffle=True)
    valloader = DataLoader(trainsets[int(cid)], batch_size=16)
    return FlowerClient(trainloader, valloader).to_client()

strategy = fl.server.strategy.FedProx(
    proximal_mu=0.01,
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=get_evaluate_fn(), 
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
