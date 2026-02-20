# EcoTorch

A lightweight, plug-and-play tool to measure the ecological impact and efficiency of your PyTorch models.

EcoTorch runs in the background while your models learn or get tested. It tracks exactly how much power your machine uses, figures out your carbon footprint based on your location, and gives you a final efficiency score. It works seamlessly across Mac, Windows, and Linux.

## Installation

You can grab the tool directly from the public Python store. Open your terminal and type:

`pip install ecotorch`


## Quick Start

You do not need to rewrite any of your existing work to use EcoTorch. Just wrap your normal learning or testing loops inside the `TrainTracker` and `EvalTracker`.

### Quick example:

```python
import torch
from ecotorch import TrainTracker, EvalTracker, Mode

# Set up your model and data
model = ...
train_loader = ... 
test_loader = ...
epoch = ...

# Wrap your train loop in the TrainTracker
with TrainTracker(epochs=epoch, model=model, train_dataloader=train_loader) as train_tracker:
    # Training logic...
    initial_loss = 2.5
    final_loss = 0.5

# Final score
score = train_tracker.calculate_efficiency_score(initial_loss=initial_loss, final_loss=final_loss)
print(f"Efficiency Score: {score}")

# You can track evaluation and inference
with EvalTracker(test_dataloader=test_loader, train_tracker=train_tracker) as eval_tracker:
    # Evaluation logic...
    acc = 0.9

# Final score
score = eval_tracker.calculate_efficiency_score(accuracy=acc)
print(f"Efficiency Score: {score}")
```

A fully implemented example is available in [testing.py](testing/testing.py)

## How It Works
When you start the tracker, it automatically:
- Finds your location: It checks where you are in the world to find out how clean your local power grid is.
- Reads the power meter: It taps directly into your machine's graphics chip or Apple brain to read the exact power drops being used.
- Does the math: When the block finishes, it calculates your total energy used (kWh), your emitted carbon (grams of CO2), and a final efficiency score based on how much your model improved versus how much energy it burned.
