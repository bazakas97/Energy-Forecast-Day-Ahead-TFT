import torch
import pandas as pd
from energypackage.transformer_training import TFT
from energypackage.data_preprocessing import prepare_inference_data
from energypackage import training_utils

def run_inference(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model creation
    input_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    forecast_length = config['forecast_length']
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

    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()

    # Loading the scalers
    scalers = training_utils.load_scalers(config['scalers_path'])

    # Preparing the data for inference
    input_data = prepare_inference_data(config['inference_csv_path'], scalers, config)

    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)
        preds = model(input_tensor)
    
    preds = preds.cpu().numpy()
    # Removing the batch dimension
    preds = preds.squeeze(0)  # now the shape is (96, 3)
    preds = preds[:, 1]
    # Creating DataFrame with appropriate columns (e.g., for quantiles 0.1, 0.5, 0.9)
    df_preds = pd.DataFrame(preds, columns=['q0.5'])
    pd.DataFrame(preds).to_csv(config['predictions_csv_path'], index=True)
    print("Inference completed and predictions saved.")
