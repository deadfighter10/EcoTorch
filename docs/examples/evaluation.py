import torch
import torch.nn as nn
from torchvision import datasets, transforms
from ecotorch import EvalTracker

# 1. Define a simple CNN for MNIST (same as in basic_training.py)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def main():
    # 2. Setup device and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=True
    )

    # Assume a pre-trained model
    model = SimpleCNN().to(device)
    model.eval()

    print("Starting tracked evaluation...")

    # 3. Use EvalTracker as a context manager
    with EvalTracker(test_dataloader=test_loader, model=model) as tracker:
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total

        # 4. Calculate and display efficiency score
        score = tracker.calculate_efficiency_score(accuracy=accuracy)
        print("\n--- Evaluation Results ---")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Total Energy Used: {tracker.used_energy} kWh")
        print(f"Total Time: {tracker.total_time} seconds")
        print(f"Efficiency Score: {score}")

if __name__ == "__main__":
    main()
