from .datahandler import DataHandler

import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading

def _transfer_optimizer_to_device(optimizer: optim.Optimizer, device: str) -> None:
    """
    Moves an existing optimizer to a specific device.

    :param optimizer: The pytorch optimizer.
    :type optimizer: optim.Optimizer
    :param device: The device we move the optimizer to.
    :type device: str
    :return:
    """
    for param in optimizer.state.values():
        # move state to the device
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparameter in param.values():
                if isinstance(subparameter, torch.Tensor):
                    subparameter.data = subparameter.data.to(device)
                    if subparameter._grad is not None:
                        subparameter._grad.data = subparameter._grad.data.to(device)

def evaluate(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: str
) -> float:
    """
    Base implementation of an accuracy-based evaluation.
    :param model: The Pytorch model for the test
    :type model: nn.Module
    :param test_loader: The pytorch DataLoader for the test dataset
    :type test_loader: torch.utils.data.DataLoader
    :param device: The device the evaluation will run on
    :type device: str
    :return: The accuracy of the model on the test dataset
    :rtype: float
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            numbers, labels = data
            numbers, labels = numbers.to(device), labels.to(device)
            outputs = model(numbers)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return round(100 * correct/total, 2)


def train(model: nn.Module,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          train_loader:torch.utils.data.DataLoader,
          epoch: int,
          device: str
) -> tuple[nn.Module, float, nn.Module, optim.Optimizer]:
    """
    Trains a neural network model using the provided dataset and optimizer.

    :param model: The neural network model to train.
    :type model: nn.Module
    :param criterion: The loss function used to compute the error
        during training.
    :type criterion: nn.Module
    :param optimizer: The optimizer used to apply computed gradients
        during backpropagation.
    :type optimizer: optim.Optimizer
    :param train_loader: The data loader that provides the training
        dataset in mini-batches.
    :type train_loader: torch.utils.data.DataLoader
    :param epoch: The number of epochs (full dataset iterations) to
        train the model for.
    :type epoch: int
    :param device: The device the training will run on
    :type device: str
    :return: A dictionary containing the trained model.
    :rtype: tuple[nn.Module, float, nn.Module, optim.Optimizer]
    """
    model = model.to(device)
    criterion = criterion.to(device)
    _transfer_optimizer_to_device(optimizer, device)

    running_loss = 0.0
    for epoch in range(epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 500 == 499:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    return model, running_loss, criterion, optimizer


class _Monitor(threading.Thread):
    def __init__(self, name, data_list: list, stop_event: threading.Event):
        super().__init__()
        self.name = name
        self.data_list = data_list
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            #TODO Implement NVIDIA and other monitoring logic
            reading = 1
            self.data_list.append(reading)
            time.sleep(1)

class Tracker:
    """
    Tracks energy usage and cost during a specific time window.

    This class is designed to monitor and collect data related to energy usage
    within a defined duration, using GPU monitoring. It provides an interface
    to perform initialization and properly handle resource cleanup through
    context management.

    :ivar energy_usage: A list that stores energy usage data collected during
        monitoring.
    :type energy_usage: list
    :ivar cost: The calculated cost based on the energy usage data.
    :type cost: float
    :ivar start_time: The timestamp capturing the start of the monitoring period.
    :type start_time: float
    :ivar end_time: The timestamp capturing the end of the monitoring period.
    :type end_time: float
    """
    def __init__(self) -> None:
        self.energy_usage = []
        self.cost = 0
        self.start_time = 0
        self.end_time = 0
        self.stop_event = threading.Event()
        self.gpu_monitor = None
        self.handler = DataHandler()

    def __enter__(self):
        self.start_time = time.time()
        self.gpu_monitor = _Monitor("GPU", self.energy_usage, self.stop_event)
        self.gpu_monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.gpu_monitor.join()
        self.end_time = time.time()
        print(f"Collected {len(self.energy_usage)} data.")
        return False

    def calculate_kwh(self):
        #TODO kWh calculation logic
        pass

    def get_co2_intensity_per_country(self, country: str):
        #TODO retrieve CO2 intensity logic
        pass

    def fetch_common_cloud_prices(self):
        #TODO Write the fetch for common cloud prices
        pass

