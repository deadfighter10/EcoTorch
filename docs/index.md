# EcoTorch

EcoTorch is a lightweight Python package designed to measure and track the ecological and financial impact of training and evaluating PyTorch models.

## Purpose

As machine learning models grow in complexity, their energy consumption and carbon footprint have become significant concerns. EcoTorch provides developers with simple tools to:
- Monitor real-time energy usage.
- Calculate carbon emissions based on local grid intensity.
- Evaluate model performance relative to its environmental cost using a specialized efficiency score.

## Key Features

- **Seamless Integration**: Track training and evaluation sessions using simple Python context managers.
- **Hardware Monitoring**: Support for NVIDIA GPUs (via NVML) and Apple Silicon (via custom SMC monitoring).
- **Global Carbon Intensity**: Automatically detects location and uses up-to-date carbon intensity data.
- **Efficiency Scoring**: Provides a mathematical score that balances model improvement with energy expenditure.

## Quick Installation

Install EcoTorch via pip:

```bash
pip install ecotorch
```

Explore the [Getting Started](getting-started.md) guide to begin tracking your models!
