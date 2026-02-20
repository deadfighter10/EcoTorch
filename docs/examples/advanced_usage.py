import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ecotorch import TrainTracker, EvalTracker

# Simple synthetic model and data
model = nn.Linear(10, 1)
data = torch.randn(100, 10)
target = torch.randn(100, 1)
dataset = TensorDataset(data, target)
loader = DataLoader(dataset, batch_size=10)

def advanced_demo():
    print("--- Advanced Usage Demo ---")

    # 1. Manually specifying a country (ISO 3-letter code)
    print("\n1. Tracking with a specific country (e.g., France - FRA)")
    with TrainTracker(model=model, epochs=1, train_dataloader=loader, country="FRA") as tracker:
        # Simulate training
        pass
    print(f"Detected/Specified Country: {tracker.country}")

    # 2. Reusing a TrainTracker for Evaluation
    print("\n2. Reusing TrainTracker for Evaluation")
    with TrainTracker(model=model, epochs=1, train_dataloader=loader) as train_tracker:
        # Simulate training
        pass
    
    # EvalTracker can take a train_tracker to inherit the model and data handler
    with EvalTracker(test_dataloader=loader, train_tracker=train_tracker) as eval_tracker:
        # Simulate evaluation
        pass
    
    # 3. Accessing detailed metrics
    print("\n3. Accessing raw metrics")
    print(f"Energy Usage List (per second): {eval_tracker.energy_usage}")
    print(f"Total Time: {eval_tracker.total_time}s")
    print(f"Used Energy: {eval_tracker.used_energy} kWh")

if __name__ == "__main__":
    advanced_demo()
