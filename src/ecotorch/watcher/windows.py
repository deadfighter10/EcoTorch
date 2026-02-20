import pynvml
from .interface import PowerMonitor

class WindowsMonitor(PowerMonitor):
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            self.device_count = 0

    def get_current_power(self) -> float:
        if self.device_count == 0:
            return 0.0

        total_power_watts = 0.0

        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)

            total_power_watts += (power_milliwatts / 1000.0)

        return total_power_watts

    def get_gpu_utilization(self):
        if self.device_count == 0:
            return "0%"

        total_util = 0.0
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            total_util += utilization.gpu

        avg_util = total_util / self.device_count
        return f"{round(avg_util, 1)}%"