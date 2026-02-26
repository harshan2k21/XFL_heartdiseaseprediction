import flwr as fl
from task import Net
from client import FlowerClient
from data_partining import partition_data
from torch.utils.data import DataLoader

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

print("🚀 Initializing Federated Heart Disease Prediction...")

# 1. Prepare Data
NUM_CLIENTS = 5
trainsets = partition_data(num_clients=NUM_CLIENTS)

def client_fn(cid: str):
    trainloader = DataLoader(trainsets[int(cid)], batch_size=16, shuffle=True)
    valloader = DataLoader(trainsets[int(cid)], batch_size=16)
    return FlowerClient(trainloader, valloader).to_client()

# 2. Define FedProx Strategy
strategy = fl.server.strategy.FedProx(
    proximal_mu=0.01,
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# 3. Start Simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
