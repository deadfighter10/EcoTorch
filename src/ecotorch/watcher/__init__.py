import sys
import platform
from typing import Type, Union, TYPE_CHECKING
from .interface import PowerMonitor

if TYPE_CHECKING:
    from .apple import Monitor as AppleMonitor

def get_monitor_class() -> Union[Type[PowerMonitor], Type["AppleMonitor"]]:
    system = platform.system()

    if system == "Darwin":
        try:
            from .apple import Monitor as AppleMonitor
            return AppleMonitor
        except ImportError:
            # Fallback or error if the C++ build failed
            raise ImportError("EcoTorch C++ extension for macOS not found.")

    elif system == "Linux":
        from .linux import LinuxMonitor
        return LinuxMonitor

    elif system == "Windows":
        from .windows import WindowsMonitor
        return WindowsMonitor

    else:
        raise OSError(f"Unsupported operating system: {system}")

Monitor = get_monitor_class()