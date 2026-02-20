"""
A lightweight package to measure the ecological and financial effect of training and evaluation of pytorch projects.
"""

from .core import TrainTracker, EvalTracker, evaluate, train, Mode
from .datahandler import DataHandler
from .geolocator import get_location, get_ip

__version__ = "0.2.2"
__author__ = "David Leonard Nagy"
__copyright__ = "Copyright 2026, David Leonard Nagy"
__credits__ = ["David Leonard Nagy"]
__license__ = "MIT"
__maintainer__ = "David Leonard Nagy"
__email__ = "nagy.david.leonard@gmail.com"
__status__ = "Development"

__all__ = [
    "TrainTracker",
    "EvalTracker",
    "DataHandler",
    "evaluate",
    "train",
    "Mode"
]