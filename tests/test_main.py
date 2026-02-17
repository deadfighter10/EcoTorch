import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import MagicMock, patch
from ecotorch.main import _transfer_optimizer_to_device, evaluate, train, Tracker

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def mock_model():
    return SimpleModel()

@pytest.fixture
def mock_loader():
    # Create simple data
    inputs = torch.randn(5, 10)
    labels = torch.randint(0, 2, (5,))
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    return loader

def test_transfer_optimizer_to_device():
    # Create a simple optimizer
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # We can't really move to cuda if not available, so maybe test with 'cpu'
    # or mock .to() calls if we want to ensure it calls it.

    # Mock optimizer state
    p = list(model.parameters())[0]
    optimizer.state[p] = {'step': 0, 'momentum_buffer': torch.zeros_like(p)}

    if torch.cuda.is_available():
        device = 'cuda'
        _transfer_optimizer_to_device(optimizer, device)
        # Check if things moved, but that requires real GPU.
    else:
        # Just run on cpu and ensure no errors
        _transfer_optimizer_to_device(optimizer, 'cpu')

def test_evaluate(mock_model, mock_loader):
    device = 'cpu'
    accuracy = evaluate(mock_model, mock_loader, device)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 100

def test_train(mock_model, mock_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mock_model.parameters(), lr=0.01)
    device = 'cpu'

    # Train for 1 epoch
    trained_model, loss, crit, opt = train(mock_model, criterion, optimizer, mock_loader, epoch=1, device=device)

    assert isinstance(trained_model, nn.Module)
    assert isinstance(loss, float)
    assert crit == criterion
    assert opt == optimizer

def test_tracker_context_manager():
    # Mock Monitor so we don't spawn threads
    with patch('ecotorch.main._Monitor') as MockMonitor:
        monitor_instance = MockMonitor.return_value

        with Tracker() as tracker:
            assert tracker.start_time != 0
            assert tracker.gpu_monitor == monitor_instance
            monitor_instance.start.assert_called_once()

        monitor_instance.join.assert_called_once()
        assert tracker.end_time != 0
        assert tracker.stop_event.is_set()

def test_tracker_methods():
    tracker = Tracker()
    # These methods are currently passing or placeholders
    tracker.calculate_kwh()
    tracker.get_co2_intensity_per_country('USA')
    tracker.fetch_common_cloud_prices()

