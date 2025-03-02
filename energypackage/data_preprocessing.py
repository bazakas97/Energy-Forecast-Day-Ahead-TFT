import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class EnergyDataset(Dataset):
    """
    PyTorch Dataset for energy consumption data.
    Contains X in the form (96, input_dim) and y in the form (96,).
    """
    def __init__(self, X, y):
        # X shape: (num_samples, 96, input_dim)
        # y shape: (num_samples, 96)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(csv_path):
    """
    Load data from CSV file.
    """
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Fill NaN in 'Energy_24h_prior' with 0 (or as needed)
    if 'Energy_24h_prior' in df.columns:
        df['Energy_24h_prior'] = df['Energy_24h_prior'].fillna(0)
    
    # Fill other NaNs if any
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df

def create_sliding_windows(df, features, target, window_size=96, stride=96):
    """
    Create sliding windows WITHOUT flattening:
    
    - Step (i): take df[i : i+window_size] as "window A" (this is the OLD day)
    - Then take df[i+window_size : i+2*window_size] as "window B" (next day)
      and we want to predict the consumption in this B.

    Here, however, because we want the features to correspond to the exact time step,
    many define that X = window A, y = Energy of window B.
    Or, as you have, X = features from B, y = Energy from B.
    
    For simplicity, we follow the same way you had, just not flattening.

    We return:
       X shape -> (num_samples, 96, len(features)+1)   # +1 for 'Energy' of the previous day, if needed
       y shape -> (num_samples, 96)
    """
    X_list = []
    y_list = []
    
    num_samples = (len(df) - window_size * 2) // stride + 1
    # Be careful not to get negative if the data is few.
    # You can use max(..., 0) if needed.

    for i in range(0, len(df) - window_size * 2 + 1, stride):
        # "window" = old day
        window = df.iloc[i : i + window_size]
        # "window_next" = next day
        window_next = df.iloc[i + window_size : i + 2*window_size]
        if len(window_next) < window_size:
            break
        
        # Input features for the next day:
        #  e.g. X_features = window_next[features]
        X_features = window_next[features].values  # shape: (96, num_features)
        
        # Energy of the previous day:
        #  if you want it as a feature, we can take it from the "window",
        #  here you seem to use 'Energy' of the previous day.
        X_energy = window['Energy'].values.reshape(-1, 1)  # shape (96,1)
        
        # Combination (features_next_day, energy_previous_day)
        X_combined = np.hstack([X_features, X_energy])  # shape (96, num_features+1)
        
        # Target: Energy of the next day
        y_target = window_next[target].values  # shape: (96,)

        X_list.append(X_combined)
        y_list.append(y_target)
    
    X_arr = np.array(X_list)  # (num_samples, 96, num_features+1)
    y_arr = np.array(y_list)  # (num_samples, 96)
    return X_arr, y_arr

def prepare_datasets(csv_path, window_size=96, stride=96, test_size=0.2, val_size=0.2):
    """
    Create training, validation, and test datasets (EnergyDataset).
    """
    df = load_data(csv_path)
    # You can add 'Energy_24h_prior' if you want explicitly.
    # Here we assume we want features=7 + 1 = 8
    # e.g. features = ['hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos','motion_detection']
    features = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']
    target = 'Energy'
    
    X, y = create_sliding_windows(df, features, target, window_size, stride)
    
    # Split (train, val, test) without shuffle (time-series)
    # Be very careful: if you want strict chronological order, do not use shuffle=True.
    # Use train_test_split with shuffle=False.
    N = len(X)
    train_end = int(N * (1 - test_size - val_size))
    val_end = int(N * (1 - test_size))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    train_dataset = EnergyDataset(X_train, y_train)
    val_dataset = EnergyDataset(X_val, y_val)
    test_dataset = EnergyDataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset, df

def create_residual_datasets(X_train, residual_train, X_val, residual_val):
    """
    Accepts X_train, residual_train (y - y_hat), 
    converts them to EnergyDataset for ARResidual training.

    X_train: (num_samples, 96, input_dim)
    residual_train: (num_samples, 96)

    We return 2 Datasets: (X_train, residual_train) and (X_val, residual_val).
    """
    train_residual_dataset = EnergyDataset(X_train, residual_train)
    val_residual_dataset = EnergyDataset(X_val, residual_val)
    return train_residual_dataset, val_residual_dataset

def get_dataloaders(csv_path, window_size=96, stride=96, test_size=0.2, val_size=0.2, batch_size=32):
    """
    Returns DataLoaders for train, val, test and the DataFrame.
    """
    train_dataset, val_dataset, test_dataset, df = prepare_datasets(
        csv_path, window_size, stride, test_size, val_size
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, df

def prepare_inference_data(csv_path, scalers, config):
    """
    Prepare data for inference.
    
    Reads the CSV, sorts the data by 'datetime' and selects the last window_size samples 
    for the columns defined in the config (e.g., the 6 temporal features and the target).
    Returns the sample in shape (1, window_size, num_features).
    
    :param csv_path: Path to the CSV file.
    :param scalers: Dictionary with scalers, if any (can be None).
    :param config: Dictionary of settings, containing 'window_size' and a list of 'inference_features'.
    :return: NumPy array with shape (1, window_size, num_features)
    """
    # Read and sort the CSV
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    # Use ffill() and bfill() instead of fillna(method=...)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    window_size = config['window_size']
    # Define the features we want for inference.
    # Make sure the config contains the key 'inference_features'
    features = config.get('inference_features', ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos', 'Energy'])
    
    # Select the corresponding columns from the DataFrame
    data = df[features].values.astype(np.float32)
    if len(data) < window_size:
        raise ValueError("Not enough data for inference")
    
    # Get the last window_size samples
    sample = data[-window_size:]  # expected shape (window_size, num_features)
    
    # If you have scalers for specific columns, you can apply them here (e.g., only for the target).
    # For the example, we skip this transformation.
    
    # Reshape to shape (1, window_size, num_features)
    sample = sample.reshape(1, window_size, len(features))
    return sample


