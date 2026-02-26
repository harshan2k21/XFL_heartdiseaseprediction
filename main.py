import flwr as fl
from task import Net
from client import FlowerClient
from data_partining import partition_data
from torch.utils.data import DataLoader
import ray
ray.init(address='auto') # Add this line
print("🚀 Initializing Federated Learning Simulation...")

# 1. Prepare Data
NUM_CLIENTS = 5
print(f"📦 Partitioning data for {NUM_CLIENTS} hospitals...")
trainsets = partition_data(num_clients=NUM_CLIENTS)

def client_fn(cid: str):
    trainloader = DataLoader(trainsets[int(cid)], batch_size=32, shuffle=True)
    valloader = DataLoader(trainsets[int(cid)], batch_size=32)
    return FlowerClient(trainloader, valloader).to_client()

# # 2. Define Strategy
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,
#     min_fit_clients=NUM_CLIENTS,
#     min_available_clients=NUM_CLIENTS,
# )

# Change your strategy to the official FedProx class
strategy = fl.server.strategy.FedProx(
    proximal_mu=0.01,  # The server now "owns" the mu value
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
)

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Return weighted average
    return {"accuracy": sum(accuracies) / sum(examples)}

# Update your strategy line
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average, # Add this line!
)

print("🏁 Starting simulation rounds...")

# 3. Start Simulation
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

print("\n✅ Simulation Complete!")
print(f"Final Accuracy Results: {history.metrics_distributed}")