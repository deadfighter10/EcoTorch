# Frequently Asked Questions (FAQ)

## Does it work on my GPU?

EcoTorch uses `pynvml` to monitor NVIDIA GPUs. If you have an NVIDIA GPU and `pynvml` installed, it should work out-of-the-box. It also supports Apple Silicon GPUs via a custom SMC monitoring extension.

## How accurate is the carbon estimate?

The carbon estimate is as accurate as the underlying power draw and the average carbon intensity data. We use historical average CO2 intensity per country. While this is a good estimate, real-time grid intensity can vary significantly.

## Can I use it without an internet connection?

Yes, but location detection will fail. You should manually specify the country code in the `TrainTracker` or `EvalTracker` constructor:

```python
tracker = TrainTracker(model, epochs, loader, country="USA")
```

If no country is specified and there's no internet, EcoTorch will default to the global average carbon intensity ("World").

## What is the C-Score?

The C-Score is a baseline factor used to normalize carbon emissions based on the complexity of the task (number of parameters and data samples). It ensures that larger models aren't unfairly penalized for doing more work, while still rewarding more efficient training and evaluation.

## Does it track CPU power?

On Apple Silicon (macOS), EcoTorch tracks both CPU and GPU power draw. On Linux and Windows, power monitoring is currently focused on NVIDIA GPUs. Support for CPU power draw via Intel RAPL on Linux/Windows is planned for future releases.
