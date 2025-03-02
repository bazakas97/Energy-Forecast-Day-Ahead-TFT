import os
import torch
from energypackage.training_functions import train_tft_model
from energypackage.transformer_training import TFT
from energypackage.data_preprocessing import get_dataloaders
from energypackage import training_utils

def train_model(config):
    csv_path = config['csv_path']
    window_size = config['window_size']
    forecast_length = config['forecast_length']
    batch_size = config['batch_size']
    epochs = config['epochs']
    output_dir = config['output_dir']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, test_loader, df = get_dataloaders(
        csv_path=csv_path,
        window_size=window_size,
        stride=96,  # example for non-overlapping windows
        test_size=config['test_size'],
        val_size=config['val_size'],
        batch_size=batch_size
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    input_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    model = TFT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        forecast_length=forecast_length,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_heads=config['num_heads'],
        quantiles=config['quantiles'],
        static_input_dim=0
    ).to(device)



    print("Starting training...")
    train_tft_model(
        model,
        train_loader,
        val_loader,
        epochs,
        output_dir,
        optimizer_lr=config['optimizer_lr'],
        quantiles=config['quantiles']
    )

    # Saving the scalers (here we use a dummy dictionary as an example)
    scalers = {}  # Replace with the actual scalers you use
    os.makedirs(os.path.dirname(config['scalers_path']), exist_ok=True)
    training_utils.save_scalers(config['scalers_path'], scalers)
    print("Training complete. Model and scalers saved.")
