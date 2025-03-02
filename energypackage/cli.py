import argparse
from energypackage.training import train_model
from energypackage.inference import run_inference
from energypackage.training_utils import load_config

def main():
    parser = argparse.ArgumentParser(description="TFT Model Training and Inference CLI")
    subparsers = parser.add_subparsers(dest='command', help='Choose operation')

    # Training για 96 ώρες
    parser_train_96 = subparsers.add_parser('train_96', help='Train model for 96 hours prediction')
    parser_train_96.add_argument('--config', type=str, default='config/config_96.yaml', help='Path to config file')

    # Training για 30 ημέρες
    parser_train_30 = subparsers.add_parser('train_30', help='Train model for 30 days prediction')
    parser_train_30.add_argument('--config', type=str, default='config/config_30.yaml', help='Path to config file')

    # Inference για 96 ώρες
    parser_infer_96 = subparsers.add_parser('infer_96', help='Run inference using the 96 hours model')
    parser_infer_96.add_argument('--config', type=str, default='config/config_96.yaml', help='Path to config file')

    # Inference για 30 ημέρες
    parser_infer_30 = subparsers.add_parser('infer_30', help='Run inference using the 30 days model')
    parser_infer_30.add_argument('--config', type=str, default='config/config_30.yaml', help='Path to config file')

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command in ['train_96', 'train_30']:
        train_model(config)
    elif args.command in ['infer_96', 'infer_30']:
        run_inference(config)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
