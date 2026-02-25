"""
A lightweight package to measure the ecological and financial effect of training and evaluation of pytorch projects.
"""

from ._geolocator import get_ip, get_location
from .core import EvalTracker, Mode, TrainTracker, evaluate, train
from .datahandler import DataHandler

__version__ = "0.2.5"
__author__ = "David Leonard Nagy"
__copyright__ = "Copyright 2026, David Leonard Nagy"
__credits__ = ["David Leonard Nagy"]
__license__ = "GNU LGPLv3"
__maintainer__ = "David Leonard Nagy"
__email__ = "nagy.david.leonard@gmail.com"
__status__ = "Development"

__all__ = ["TrainTracker", "EvalTracker", "DataHandler", "evaluate", "train", "Mode"]
