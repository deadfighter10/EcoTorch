from .datahandler import DataHandler
from .watcher import Monitor

import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading
from enum import Enum, auto
import math
from abc import ABC, abstractmethod

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
    for e in range(epoch):
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
                print(f'[{e + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
    return model, running_loss/500, first_loss, criterion, optimizer


class ModeError(Exception):
    pass


class Mode(Enum):
    TRAIN = auto()
    EVAL = auto()
    OTHER = auto()

class Tracker(ABC):
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
        mode: Mode,
        model: nn.Module,
        epochs: int = None,
        train_dataloader: torch.utils.data.DataLoader = None,
        test_dataloader: torch.utils.data.DataLoader = None,
        data_handler: DataHandler = None,
        country: str = None
    ) -> None:
        self.energy_usage = []
        self.cost = 0
        self.total_time = 0

        self._gpu_monitor = None
        if data_handler is None:
            self._handler = DataHandler()
        else:
            self._handler = data_handler
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
        self._country = country

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

        print(f"The block emitted {self._co2_emission}g CO2")
        print(f"The block took {self.total_time} second to execute.")
        print(f"The block used {self.used_energy} kwh to execute.")
        return False

    @abstractmethod
    def calculate_efficiency_score(self, accuracy: float = None, initial_loss: float = None, final_loss: float = None) -> float:
        """
        Calculate the efficiency score of a model.

        :param initial_loss: The first loss in the training.
        :param final_loss: The final loss in the training.
        :param accuracy: Accuracy of the model.
        :return: The efficiency score.
        """
        pass

    @property
    @abstractmethod
    def _calculate_c_score(self) -> int:
        pass

    @property
    def _calculate_kwh(self) -> float:
        if len(self.energy_usage) == 0:
            return 0.0
        return sum(self.energy_usage) / 3600000

    @property
    def _co2_emission(self) -> float:
        if self._country is not None: return round(self._calculate_kwh * self._handler.get_intensity(self._country), 4)
        return round(self._calculate_kwh * self._handler.get_intensity(), 4)

    @property
    def _data_handler(self) -> DataHandler:
        return self._handler

    @property
    def used_energy(self) -> float:
        return round(self._calculate_kwh, 4)


class TrainTracker(Tracker):
    def __init__(self,
        model: nn.Module,
        epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        country: str = None
    ) -> None:
        self._model = model
        self._epochs = epochs
        self._train_dataloader = train_dataloader
        self._country = country
        super().__init__(Mode.TRAIN, model, epochs=epochs, train_dataloader=train_dataloader, country=country)

    def calculate_efficiency_score(self, accuracy: float = None, initial_loss: float = None, final_loss: float = None) -> float:
        """
        Calculate the efficiency score of a model.

        :param initial_loss: The first loss in the training.
        :param final_loss: The final loss in the training.
        :param accuracy: Accuracy of the model.
        :return: The efficiency score.
        """

        if accuracy is not None and not (0.0 <= accuracy <= 1.0):
            raise ValueError("Accuracy must be between 0 and 1 (inclusive).")

        if not self._is_closed:
            raise RuntimeError(
                "The Tracker did not finish tracking; call this only after the 'with' block completes."
            )

        if initial_loss is None:
            raise ValueError("Initial loss cannot be None in train mode!")
        if final_loss is None:
            raise ValueError("Final loss cannot be None in train mode!")

        _c_score = self._calculate_c_score
        if initial_loss == 0.0:
            _progress = 0.0
        else:
            _progress = max(0.0, 1.0 - (final_loss / initial_loss))
        _emission = self._co2_emission
        _penalty = math.exp(-1.0 * (_emission / _c_score))
        score = _progress * _penalty

        return round(score, 4)

    @property
    def _calculate_c_score(self) -> int:
        P = sum(p.numel() for p in self._model.parameters())
        D = getattr(self._dataloader.dataset, '__len__', lambda: 0)() * self._epochs
        return 6 * P * D

    @property
    def country(self) -> str:
        return self._country

    @property
    def model(self) -> nn.Module:
        return self._model


class EvalTracker(Tracker):
    def __init__(self,
        test_dataloader: torch.utils.data.DataLoader,
        train_tracker: TrainTracker = None,
        model: nn.Module = None,
        country: str = None,
    ) -> None:
        if train_tracker:
            if country is not None:
                self._country = country
            else:
                self._country = train_tracker.country
            super().__init__(Mode.EVAL, train_tracker.model, test_dataloader=test_dataloader, data_handler=train_tracker._data_handler, country=self._country)
        else:
            if model is None:
                raise ValueError("Please provide a model or TrainTracker!")

            super().__init__(Mode.EVAL, model, test_dataloader=test_dataloader, country=country)

    def calculate_efficiency_score(self, accuracy: float = None, initial_loss: float = None, final_loss: float = None) -> float:
        """
        Calculate the efficiency score of a model.

        :param initial_loss: The first loss in the training.
        :param final_loss: The final loss in the training.
        :param accuracy: Accuracy of the model.
        :return: The efficiency score.
        """

        if accuracy is not None and not (0.0 <= accuracy <= 1.0):
            raise ValueError("Accuracy must be between 0 and 1 (inclusive).")

        if not self._is_closed:
            raise RuntimeError(
                "The Tracker did not finish tracking; call this only after the 'with' block completes."
            )

        if accuracy is None:
            raise ValueError("Accuracy cannot be None in eval mode!")

        _c_score = self._calculate_c_score
        _emission = self._co2_emission
        _penalty = math.exp(-1.0 * (_emission / _c_score))
        score = accuracy * _penalty

        return round(score, 4)

    @property
    def _calculate_c_score(self) -> int:
        P = sum(p.numel() for p in self._model.parameters())
        D = getattr(self._dataloader.dataset, '__len__', lambda: 0)()
        return 2 * P * D

    @property
    def country(self) -> str:
        return self._country


class _Monitor(threading.Thread):
    def __init__(self,
        name,
        data_list: list,
        stop_event: threading.Event
    ) -> None:
        super().__init__()
        self._name = name
        self._data_list = data_list
        self._stop_event = stop_event
        self._watcher = Monitor()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._data_list.append(self._watcher.get_current_power())
            self._stop_event.wait(1)

    @property
    def data_list(self) -> list:
        return self._data_list
