# EcoTorch

[![PyPI version](https://img.shields.io/pypi/v/ecotorch.svg)](https://pypi.org/project/ecotorch/)
[![Python versions](https://img.shields.io/pypi/pyversions/ecotorch.svg)](https://pypi.org/project/ecotorch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EcoTorch is a lightweight, plug-and-play Python package designed to measure and track the ecological and financial impact of training and evaluating PyTorch models.

## Key Features

- **Seamless Integration**: Track training and evaluation sessions using simple Python context managers.
- **Hardware Monitoring**: Support for NVIDIA GPUs (via NVML) and Apple Silicon (via custom SMC monitoring).
- **Global Carbon Intensity**: Automatically detects location and uses up-to-date carbon intensity data.
- **Efficiency Scoring**: Provides a specialized score that balances model improvement with environmental cost.

## Installation

Install EcoTorch via pip:

```bash
pip install ecotorch
```

## Quick Start

You don't need to rewrite your existing code. Just wrap your training or evaluation loops with `TrainTracker` or `EvalTracker`.

```python
import torch
from ecotorch import TrainTracker, EvalTracker

model = ...
train_loader = ...
epochs = 10

# Track Training
with TrainTracker(model=model, epochs=epochs, train_dataloader=train_loader) as tracker:
    # Your training loop here
    initial_loss = 2.5
    final_loss = 0.5

# Calculate efficiency score
score = tracker.calculate_efficiency_score(initial_loss=initial_loss, final_loss=final_loss)
print(f"Training Efficiency Score: {score}")
```

## Documentation

For full documentation, including a getting started guide, API reference, and detailed methodology, please see the [docs/](docs/) directory:

- [Introduction](docs/index.md)
- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Methodology](docs/methodology.md)
- [Examples](docs/examples/)
- [FAQ](docs/faq.md)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md) and [Code of Conduct](docs/code-of-conduct.md).

## License

EcoTorch is released under the MIT License.
