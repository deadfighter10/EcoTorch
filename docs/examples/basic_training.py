import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from ecotorch import TrainTracker

# 1. Define a simple CNN for MNIST
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
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = 1

    print("Starting tracked training...")

    # 3. Use TrainTracker as a context manager
    with TrainTracker(model=model, epochs=epochs, train_dataloader=train_loader) as tracker:
        initial_loss = None
        
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if initial_loss is None:
                    initial_loss = loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
        
        final_loss = loss.item()

        # 4. Calculate and display efficiency score
        score = tracker.calculate_efficiency_score(initial_loss=initial_loss, final_loss=final_loss)
        print("\n--- Training Results ---")
        print(f"Total Energy Used: {tracker.used_energy} kWh")
        print(f"Total Time: {tracker.total_time} seconds")
        print(f"Efficiency Score: {score}")

if __name__ == "__main__":
    main()
