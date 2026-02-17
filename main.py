import torch
import torch.nn as nn


def evaluate(model: nn.Module, test_loader: torch.utils.data.DataLoader) -> float:
    """
    Base implementation of an accuracy evaluation
    :param model: The Pytorch model for the test
    :param test_loader: The pytorch DataLoader for the test dataset
    :return: The accuracy of the model on the test dataset
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            numbers, labels = data
            outputs = model(numbers)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct/total

