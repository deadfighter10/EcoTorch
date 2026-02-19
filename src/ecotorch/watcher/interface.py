from abc import ABC, abstractmethod

class PowerMonitor(ABC):
    """
    The blueprint class for the platform specific subclasses.
    """

    @abstractmethod
    def get_current_power(self):
        """Returns the power usage in Watts."""
        pass

    @abstractmethod
    def get_gpu_utilization(self):
        """Returns GPU utilization as a percentage."""
        pass