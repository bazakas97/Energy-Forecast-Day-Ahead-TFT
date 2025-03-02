# Energy-Forecast-Day-Ahead-TFT


This package provides tools for energy consumption forecasting using advanced deep learning models (e.g., Temporal Fusion Transformer or custom implementations). It includes functionalities for:

- **Training** deep learning models using sliding windows.
- **Inference** with the trained models.
- Data preprocessing and creation of DataLoaders.
- A command-line interface (CLI) for easy execution.

## Project Structure

```
energyforecast/ 
├── config/
│ ├── config_96.yaml # Settings for 96-hour prediction 
│ └── config_30.yaml # Settings for 30-day prediction 
├── data/ 
│ ├── updated_dataset_threshold_1_5.csv # Data file for training 
│ ├── inference_input_96.csv # Inference data for 96-hour prediction 
│ └── inference_input_30.csv # Inference data for 30-day prediction 
├── energypackage/ # Python modules for the package 
│ ├── init.py 
│ ├── cli.py # CLI for running training/inference 
│ ├── training.py # Training logic 
│ ├── inference.py # Inference logic 
│ ├── training_functions.py # Main training loop and custom loss functions 
│ ├── training_utils.py # Utility functions (e.g., load_config, save_scalers) 
│ ├── data_preprocessing.py # Functions to create DataLoaders and preprocess data 
│ └── transformer_training.py # Model definition (e.g., TFT) ├── requirements.txt # Project dependencies 
└── setup.py # Package installation settings
```

## Installation

### 1. Local Installation

After cloning the repository, navigate to the project root and run:

```bash
pip install .
```
Alternatively, you can build the distribution and install the wheel:

```bash
python setup.py sdist bdist_wheel
pip install dist/energy_forecast-0.1-py3-none-any.whl
```

### 2. Installing CPU-only PyTorch

If you do not require GPU support and want a smaller, CPU-only version of PyTorch, install the package using:

```bash
pip install . --extra-index-url https://download.pytorch.org/whl/cpu
```
This ensures that pip retrieves the CPU-only build of PyTorch.

## Usage

The package provides a CLI entry point named `energy_cli.` You can run it from the command line as follows:

### Training
To train the model for a 96-hour prediction:
```bash
energy_cli train_96 --config config/config_96.yaml
```
To train for a 30-day prediction:
```bash
energy_cli train_30 --config config/config_30.yaml
```
### Inference
To perform inference for 96 hours:
```bash
energy_cli infer_96 --config config/config_96.yaml
```
To perform inference for 30 days:
```bash
energy_cli infer_30 --config config/config_30.yaml
```

## Configuration Files
The configuration files (e.g., config_96.yaml) include parameters such as:

- csv_path: Path to the CSV data file.
- window_size: Number of time steps for input.
- forecast_length: Number of time steps to predict.
- batch_size and epochs
- input_dim, hidden_dim, num_layers, dropout, num_heads
- quantiles: Quantiles used for prediction (e.g., [0.1, 0.5, 0.9]).
- Paths for saving the model and scalers.

Adjust these values according to your project requirements.

##Contributing
If you have questions or would like to contribute, please open an issue or submit a pull request on GitHub.

## Docker
If you prefer to containerize the package, create a Dockerfile in the project root with the following content:
``` bash
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install .

ENTRYPOINT ["energy_cli"]
```
Then build and run the container:
```bash
docker build -t energy_forecast .
docker run energy_forecast infer_96 --config config/config_96.yaml
```
