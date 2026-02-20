import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch
from ecotorch.core import _transfer_optimizer_to_device, evaluate, train, TrainTracker, EvalTracker


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

    # Mock optimizer state
    p = list(model.parameters())[0]
    optimizer.state[p] = {'step': 0, 'momentum_buffer': torch.zeros_like(p)}

    # Just ensure no exceptions for CPU
    _transfer_optimizer_to_device(optimizer, 'cpu')


def test_evaluate(mock_model, mock_loader):
    device = 'cpu'
    accuracy = evaluate(mock_model, mock_loader, device)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_train(mock_model, mock_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mock_model.parameters(), lr=0.01)
    device = 'cpu'

    # Train for 1 epoch
    trained_model, loss, first_loss, crit, opt = train(
        mock_model, criterion, optimizer, mock_loader, epoch=1, device=device
    )

    assert isinstance(trained_model, nn.Module)
    assert isinstance(loss, float)
    assert isinstance(first_loss, float)
    assert crit == criterion
    assert opt == optimizer


def test_traintracker_context_and_score(mock_model, mock_loader):
    with patch('ecotorch.core._Monitor') as MockMonitor, \
         patch('ecotorch.datahandler.DataHandler.get_intensity', return_value=0.5):

        with TrainTracker(model=mock_model, epochs=1, train_dataloader=mock_loader, country='USA') as tracker:
            assert tracker._start_time != 0
            MockMonitor.return_value.start.assert_called_once()
            # Simulate power samples in Watts collected each second
            tracker.energy_usage = [100.0, 120.0]

        MockMonitor.return_value.join.assert_called_once()
        assert tracker._is_closed
        # Score in [0,1]
        score = tracker.calculate_efficiency_score(initial_loss=2.0, final_loss=1.0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_evaltracker_context_and_score(mock_model, mock_loader):
    with patch('ecotorch.core._Monitor') as MockMonitor, \
         patch('ecotorch.datahandler.DataHandler.get_intensity', return_value=0.5):

        # Reuse TrainTracker's model/handler via constructor
        train_tracker = TrainTracker(model=mock_model, epochs=1, train_dataloader=mock_loader, country='USA')
        with EvalTracker(test_dataloader=mock_loader, train_tracker=train_tracker) as et:
            MockMonitor.return_value.start.assert_called_once()
            et.energy_usage = [50.0]
        MockMonitor.return_value.join.assert_called_once()

        score = et.calculate_efficiency_score(accuracy=0.85)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert et.country == 'USA'


def test_validation_errors(mock_model, mock_loader):
    with patch('ecotorch.core._Monitor'), \
         patch('ecotorch.datahandler.DataHandler.get_intensity', return_value=0.5):

        # Calling before context exit should raise
        tt = TrainTracker(model=mock_model, epochs=1, train_dataloader=mock_loader)
        with pytest.raises(RuntimeError):
            tt.calculate_efficiency_score(initial_loss=1.0, final_loss=0.5)

        # After exit, invalid accuracy bounds should raise
        with TrainTracker(model=mock_model, epochs=1, train_dataloader=mock_loader) as tt2:
            tt2.energy_usage = [10.0]
        with pytest.raises(ValueError):
            tt2.calculate_efficiency_score(initial_loss=1.0, final_loss=0.5, accuracy=1.5)

        # EvalTracker: accuracy is required and must be in [0,1]
        with EvalTracker(test_dataloader=mock_loader, model=mock_model) as et:
            et.energy_usage = [10.0]
        with pytest.raises(ValueError):
            et.calculate_efficiency_score(accuracy=None)
        with pytest.raises(ValueError):
            et.calculate_efficiency_score(accuracy=-0.1)
