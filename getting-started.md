# Getting Started with EcoTorch

EcoTorch helps you measure the environmental impact of your PyTorch models. This guide will help you install EcoTorch and use it in your project.

## Installation

EcoTorch can be installed via `pip`:

```bash
pip install ecotorch
```

On Linux or Windows with NVIDIA GPUs, `pynvml` will be used for energy measurement. On macOS with Apple Silicon, EcoTorch uses a custom C++ extension for hardware monitoring.

## Minimal Working Example

This example demonstrates how to use `TrainTracker` and `EvalTracker` with a simple MNIST classifier.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from ecotorch import TrainTracker, EvalTracker

# 1. Define a simple model
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

# 2. Setup data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000, shuffle=True)

model = SimpleCNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
epochs = 1

# 3. Track Training
with TrainTracker(model=model, epochs=epochs, train_dataloader=train_loader) as tracker:
    initial_loss = None
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if initial_loss is None: initial_loss = loss.item()
    final_loss = loss.item()
    
    # Calculate efficiency score based on loss reduction
    score = tracker.calculate_efficiency_score(initial_loss=initial_loss, final_loss=final_loss)
    print(f"Training Efficiency Score: {score}")

# 4. Track Evaluation
with EvalTracker(test_dataloader=test_loader, model=model) as eval_tracker:
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    
    # Calculate efficiency score based on accuracy
    eval_score = eval_tracker.calculate_efficiency_score(accuracy=accuracy)
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    print(f"Evaluation Efficiency Score: {eval_score}")
```

## Understanding the Output

When a `with` block for a tracker completes, EcoTorch prints basic metrics:

- **CO2 Emissions**: The estimated carbon dioxide equivalents (in grams) emitted during the session, calculated from energy usage and your location's grid intensity.
- **Execution Time**: Total wall-clock time in seconds.
- **Energy Used**: Total energy consumed by the hardware in kWh.

The **Efficiency Score** is a metric between 0 and 1 (usually) that combines model improvement (loss reduction or accuracy) with environmental cost. A higher score indicates a more efficient model development or evaluation process.
