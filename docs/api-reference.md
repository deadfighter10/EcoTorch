# API Reference

This page provides detailed documentation for the public classes and functions of EcoTorch.

## TrainTracker

`TrainTracker` is a context manager used to monitor the environmental impact of training a PyTorch model.

### Constructor

`TrainTracker(model, epochs, train_dataloader, country=None)`

- **`model`** (`torch.nn.Module`): The PyTorch model being trained.
- **`epochs`** (`int`): Total number of epochs for the training run.
- **`train_dataloader`** (`torch.utils.data.DataLoader`): The dataloader used for training data.
- **`country`** (`str`, optional): ISO 3-letter country code or full country name. If `None`, EcoTorch will attempt to detect the location automatically.

### Methods

#### `calculate_efficiency_score(initial_loss, final_loss, accuracy=None)`
Calculates the efficiency score for the training session.
- **`initial_loss`** (`float`): The loss value at the start of training. (Required for training mode)
- **`final_loss`** (`float`): The loss value at the end of training. (Required for training mode)
- **`accuracy`** (`float`, optional): Final accuracy (0.0 to 1.0).
- **Returns**: `float` - The efficiency score (rounded to 4 decimal places).

### Properties
- **`used_energy`**: Returns the total energy used in kWh (rounded to 4 decimal places).
- **`total_time`**: Returns the total execution time in seconds.
- **`country`**: Returns the ISO-3 code of the country being used for carbon intensity.

---

## EvalTracker

`EvalTracker` is a context manager used to monitor the environmental impact of evaluating a PyTorch model.

### Constructor

`EvalTracker(test_dataloader, train_tracker=None, model=None, country=None)`

- **`test_dataloader`** (`torch.utils.data.DataLoader`): The dataloader used for evaluation data.
- **`train_tracker`** (`TrainTracker`, optional): An instance of a `TrainTracker`. If provided, it will reuse the same model and data handler.
- **`model`** (`torch.nn.Module`, optional): The PyTorch model being evaluated (required if `train_tracker` is not provided).
- **`country`** (`str`, optional): ISO 3-letter country code or full country name.

### Methods

#### `calculate_efficiency_score(accuracy)`
Calculates the efficiency score for the evaluation session.
- **`accuracy`** (`float`): The accuracy achieved during evaluation (0.0 to 1.0). (Required for evaluation mode)
- **Returns**: `float` - The efficiency score.

### Properties
- **`used_energy`**: Returns the total energy used in kWh.
- **`total_time`**: Returns the total execution time in seconds.

---

## Mode

An enumeration representing the tracking mode.

- **`Mode.TRAIN`**: Used for training sessions.
- **`Mode.EVAL`**: Used for evaluation sessions.
- **`Mode.OTHER`**: General tracking.

---

## DataHandler

Internal class used to manage CO2 intensity data and location detection.

### Methods

#### `get_intensity(country=None)`
Returns the CO2 intensity (gCO2/kWh) for the specified country or the detected location.

#### `countries` (Property)
Returns a sorted list of all countries for which CO2 intensity data is available.
