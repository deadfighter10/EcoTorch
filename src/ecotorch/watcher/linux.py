from .interface import PowerMonitor

class LinuxMonitor(PowerMonitor):
    def __init__(self):
        pass

    def get_current_power(self):
        """Returns the power usage in Watts."""
        pass

    def get_gpu_utilization(self):
        """Returns GPU utilization as a percentage."""
        pass