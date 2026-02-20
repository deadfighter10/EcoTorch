import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os

from ecotorch import evaluate, train, Mode, TrainTracker, EvalTracker

# Deletable, I mostly run my codes in Terminal, so it is cleaner from me to run the tests this way
os.system('clear' if os.name != "nt" else 'cls')


EPOCH = 5
BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")


transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = TestNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

with TrainTracker(epochs=EPOCH, model=net, train_dataloader=train_loader) as train_tracker:
    net, final_loss, first_loss, _, _ = train(net, criterion, optimizer, train_loader, EPOCH, device)
print(train_tracker.calculate_efficiency_score(initial_loss=first_loss, final_loss=final_loss))

print("Finished training")

with EvalTracker(test_dataloader=test_loader, train_tracker=train_tracker) as eval_tracker:
    acc = evaluate(net, test_loader, device)

print(eval_tracker.calculate_efficiency_score(accuracy=acc))

print(f"Accuracy: {acc*100}%")