"""
A lightweight package to measure the ecological and financial effect of training and evaluation of pytorch projects.
"""

from .core import Tracker, evaluate, train, Mode
from .datahandler import DataHandler

__version__ = "0.1.0"
__author__ = "David Leonard Nagy"
__copyright__ = "Copyright 2026, David Leonard Nagy"
__credits__ = ["David Leonard Nagy"]
__license__ = "MIT"
__maintainer__ = "David Leonard Nagy"
__email__ = "nagy.david.leonard@gmail.com"
__status__ = "Development"

__all__ = [
    "Tracker",
    "evaluate",
    "train"
]