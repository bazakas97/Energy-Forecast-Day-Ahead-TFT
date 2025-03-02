import pickle
import yaml

def save_scalers(path, scalers):
    with open(path, 'wb') as f:
        pickle.dump(scalers, f)

def load_scalers(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
