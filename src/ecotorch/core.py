from .datahandler import DataHandler

import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading
from enum import Enum, auto
import math

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

    return round(correct/total, 4)


def train(model: nn.Module,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          train_loader:torch.utils.data.DataLoader,
          epoch: int,
          device: str
) -> tuple[nn.Module, float, float, nn.Module, optim.Optimizer]:
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
    first_loss = 0.0
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
            if first_loss == 0.0:
                first_loss = loss.item()

            if i % 500 == 499:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
    return model, running_loss/500, first_loss, criterion, optimizer

class ModeError(Exception):
    pass

class Mode(Enum):
    TRAIN = auto()
    EVAL = auto()
    OTHER = auto()

class _Monitor(threading.Thread):
    def __init__(self, name, data_list: list, stop_event: threading.Event):
        super().__init__()
        self._name = name
        self._data_list = data_list
        self._stop_event = stop_event

    def run(self):
        start_watts = 25.0
        while not self._stop_event.is_set():
            #TODO Implement NVIDIA and other monitoring logic
            #This is a mock logic to simulating the thermal throttling on a mac m4 air
            current_watts = max(12.0, start_watts - (len(self._data_list) * 0.1))
            kw_reading = current_watts / 1000.0
            self._data_list.append(kw_reading)
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
    :ivar _start_time: The timestamp capturing the start of the monitoring period.
    :type _start_time: float
    :ivar _end_time: The timestamp capturing the end of the monitoring period.
    :type _end_time: float
    """
    def __init__(self,
        mode: Mode, model: nn.Module,
        epochs: int = None,
        train_dataloader: torch.utils.data.DataLoader = None,
        test_dataloader: torch.utils.data.DataLoader = None,
    ) -> None:
        if mode == Mode.TRAIN and train_dataloader is None:
            raise ValueError("Train dataloader cannot be None in train mode.")
        if mode == Mode.EVAL and test_dataloader is None:
            raise ValueError("Test dataloader cannot be None in test mode.")
        if mode == Mode.OTHER:
            raise NotImplementedError("No other mode is implemented yet, please use train or eval.")

        self.energy_usage = []
        self.cost = 0
        self.total_time = 0

        self._gpu_monitor = None
        self._handler = DataHandler()
        self._mode = mode
        self._is_closed = False
        self._stop_event = threading.Event()
        self._start_time = 0
        self._end_time = 0
        self._model = model
        if self._mode == Mode.TRAIN:
            self._dataloader = train_dataloader
            self._epochs = epochs
        elif self._mode == Mode.EVAL:
            self._dataloader = test_dataloader
            self._epochs = 0

    def __enter__(self):
        self._start_time = time.time()
        self._gpu_monitor = _Monitor("GPU", self.energy_usage, self._stop_event)
        self._gpu_monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._gpu_monitor.join()

        self._end_time = time.time()
        self.total_time = round(self._end_time - self._start_time, 4)
        self._is_closed = True

        print(f"The block emitted {self._get_co2_emission}g CO2")
        print(f"The block took {self.total_time} second to execute.")
        return False

    def calculate_efficiency_score(self, accuracy: float = None, initial_loss: float = None, final_loss: float = None) -> float:
        """
        Calculate the efficiency score of a model.

        :param initial_loss: The first loss in the training.
        :param final_loss: The final loss in the training.
        :param accuracy: Accuracy of the model.
        :return: The efficiency score.
        """
        # if self._mode != Mode.EVAL:
        #     raise ModeError("Tracker needs to be in EVAL mode to calculate efficiency score!")

        if accuracy is not None and not (0.0 <= accuracy <= 1.0):
            raise ValueError("Accuracy must be between 0 and 1 (inclusive).")

        if not self._is_closed:
            raise RuntimeError(
                "The Tracker did not finish tracking; call this only after the 'with' block completes."
            )

        if self._mode == Mode.TRAIN:
            if initial_loss is None:
                raise ValueError("Initial loss cannot be None in train mode!")
            if final_loss is None:
                raise ValueError("Final loss cannot be None in train mode!")
            _c_score = self._calculate_c_score_training
            _progress = max(0.0, 1.0 - (final_loss / initial_loss))
            _emission = self._get_co2_emission
            _penalty = math.exp(-1.0 * (_emission / _c_score))
            score = _progress * _penalty

        elif self._mode == Mode.EVAL:
            if accuracy is None:
                raise ValueError("Accuracy cannot be None in eval mode!")
            _c_score = self._calculate_c_score_eval
            _emission = self._get_co2_emission
            _penalty = math.exp(-1.0 * (_emission / _c_score))
            score = accuracy * _penalty

        else:
            raise RuntimeError("Failed to calculate efficiency score: unsupported mode.")

        return round(score, 4)

    @property
    def _calculate_kwh(self) -> float:
        if len(self.energy_usage) == 0:
            return 0.0
        return sum(self.energy_usage) / 3600

    @property
    def _calculate_c_score_training(self) -> int:
        P = sum(p.numel() for p in self._model.parameters())
        D = getattr(self._dataloader.dataset, '__len__', lambda: 0)() * self._epochs
        return 6 * P * D

    @property
    def _calculate_c_score_eval(self) -> int:
        P = sum(p.numel() for p in self._model.parameters())
        D = getattr(self._dataloader.dataset, '__len__', lambda: 0)()
        return 2 * P * D

    @property
    def _get_co2_emission(self) -> float:
        return round(self._calculate_kwh * self._handler.get_intensity(), 4)

    def _fetch_common_cloud_prices(self):
        #TODO Write the fetch for common cloud prices
        pass

