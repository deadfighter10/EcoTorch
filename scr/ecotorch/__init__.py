"""
A lightweight package to measure the ecological and financial effect of training and evaluation of pytorch projects.
"""

from .main import Tracker, evaluate, train

__version__ = "0.0.1"
__author__ = "Leo Nagy"

__all__ = [
    "Tracker",
    "evaluate",
    "train"
]