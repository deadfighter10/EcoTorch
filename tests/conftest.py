import sys
import os
import types

# Ensure package source is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Stub macOS apple watcher extension so ecotorch.watcher can import on Darwin
APPLE_MODULE = 'ecotorch.watcher.apple'
if APPLE_MODULE not in sys.modules:
    apple_mod = types.ModuleType(APPLE_MODULE)

    class Monitor:
        def __init__(self, *args, **kwargs):
            pass

        def get_current_power(self):
            return 0.0

        def get_gpu_utilization(self):
            return "0%"

    apple_mod.Monitor = Monitor
    sys.modules[APPLE_MODULE] = apple_mod

# Stub external runtime-only deps used in geolocator to avoid import errors in tests
if 'geoip2fast' not in sys.modules:
    geoip_mod = types.ModuleType('geoip2fast')

    class GeoIP2Fast:  # minimal stub, tests will patch methods as needed
        def lookup(self, ip):
            class _R:
                country_name = 'World'
            return _R()

    geoip_mod.GeoIP2Fast = GeoIP2Fast
    sys.modules['geoip2fast'] = geoip_mod

if 'pycountry' not in sys.modules:
    pc_mod = types.ModuleType('pycountry')

    class _Countries:
        def search_fuzzy(self, name):
            return []

    pc_mod.countries = _Countries()
    sys.modules['pycountry'] = pc_mod

# Stub pynvml to allow importing Linux/Windows monitors if needed
if 'pynvml' not in sys.modules:
    pynvml_mod = types.ModuleType('pynvml')

    class NVMLError(Exception):
        pass

    def nvmlInit():
        return None

    def nvmlDeviceGetCount():
        return 0

    def nvmlDeviceGetHandleByIndex(i):
        return None

    def nvmlDeviceGetPowerUsage(handle):
        return 0

    class _Util:
        gpu = 0

    def nvmlDeviceGetUtilizationRates(handle):
        return _Util()

    pynvml_mod.NVMLError = NVMLError
    pynvml_mod.nvmlInit = nvmlInit
    pynvml_mod.nvmlDeviceGetCount = nvmlDeviceGetCount
    pynvml_mod.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    pynvml_mod.nvmlDeviceGetPowerUsage = nvmlDeviceGetPowerUsage
    pynvml_mod.nvmlDeviceGetUtilizationRates = nvmlDeviceGetUtilizationRates

    sys.modules['pynvml'] = pynvml_mod
