# Methodology

EcoTorch aims to provide an accurate estimate of the environmental impact of PyTorch workloads. This page explains the underlying calculations and the data sources used.

## Energy Measurement

EcoTorch measures the power consumption of the hardware during the tracking period.

- **Linux & Windows (NVIDIA GPUs)**: EcoTorch uses the `pynvml` library (Python bindings for the NVIDIA Management Library - NVML) to access the power consumption of all available GPUs. It polls the GPU power usage (in Watts) every second and integrates it over time to calculate total energy in kilowatt-hours (kWh).
- **macOS (Apple Silicon)**: On macOS, EcoTorch uses a custom C++ extension that interacts with the Apple SMC (System Management Controller) to retrieve real-time power draw for both the CPU and GPU.
- **CPU Energy**: On Linux/Windows, CPU energy measurement is currently limited and typically relies on platform-specific libraries if available. Future versions will include more robust CPU monitoring (e.g., via Intel RAPL).

## Carbon Footprint Calculation

The carbon footprint of an AI model depends on the energy used and the carbon intensity of the electricity grid at the time and location of execution.

### Formulas

- **Energy (kWh)**: $\text{Total Energy (kWh)} = \frac{\sum \text{Power Readings (Watts)}}{3600000}$
- **CO2 Emissions (g)**: $\text{CO2 Emissions} = \text{Total Energy (kWh)} \times \text{Carbon Intensity (gCO2/kWh)}$

### Data Source

EcoTorch uses data from Ember's electricity data to determine carbon intensity. A local CSV file (`co2intensity.csv`) is included with the package, containing historical CO2 intensity values for various countries.

### Location Detection

By default, EcoTorch uses the public IP address of the machine to determine the country (via a geoip database). This country is then matched against the included CO2 intensity data. Users can also manually specify a country code.

## Efficiency Score

EcoTorch introduces an **Efficiency Score** to help researchers evaluate the environmental cost of model performance.

### Training Efficiency Score

For training, the score is calculated as:
$\text{Score} = (1 - \frac{\text{Final Loss}}{\text{Initial Loss}}) \times e^{-\frac{\text{CO2 Emissions}}{\text{C-Score}}}$

Where:
- **Progress** ($1 - \frac{\text{Final Loss}}{\text{Initial Loss}}$) measures the model's relative improvement.
- **Penalty** ($e^{-\frac{\text{CO2 Emissions}}{\text{C-Score}}}$) is an exponential decay factor based on carbon emissions.
- **C-Score** ($6 \times \text{Parameters} \times \text{Training Samples}$) is a baseline factor related to the model's theoretical complexity and training scale.

### Evaluation Efficiency Score

For evaluation, the score is simpler:
$\text{Score} = \text{Accuracy} \times e^{-\frac{\text{CO2 Emissions}}{\text{C-Score}}}$

Where:
- **C-Score** for evaluation is $2 \times \text{Parameters} \times \text{Testing Samples}$.

## Assumptions and Limitations

1. **GPU-centric Monitoring**: In this version, power measurement focuses primarily on GPU consumption. CPU and peripheral power draw are included where supported (macOS) but may be less accurate on other platforms.
2. **Static Carbon Intensity**: The current version uses historical yearly average data for carbon intensity. Real-time fluctuations in the grid are not captured yet.
3. **Network Connection**: Geolocation requires an internet connection to determine the public IP address. If the machine is offline, it defaults to the "World" average carbon intensity.
