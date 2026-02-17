import torch
import torch.nn as nn
import torch.optim as optim

def _transfer_optimizer_to_device(optim: optim.Optimizer, device: str) -> None:
    """
    Moves an existing optimizer to a specific device.

    :param optim: The pytorch optimizer.
    :type optim: optim.Optimizer
    :param device: The device we move the optimizer to.
    :type device: str
    :return:
    """
    for param in optim.state.values():
        # move state to the device
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

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

    return 100 * correct/total


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
