import os
import yaml
import argparse
from engine.trainer import Trainer
from utils.common import seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description='Face Recognition Training')
    parser.add_argument('--config', type=str, default='config/train_resnet50.yaml',
                        help='path to config file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='device to use')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from latest checkpoint in save_dir')
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = args.config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = config.get('seed', 42)
    seed_everything(seed, deterministic=config.get('deterministic', True))

    config['device'] = args.device
    if 'training' in config and args.resume:
        config['training']['resume'] = True

    # Initialize Trainer
    trainer = Trainer(config)
    
    # Start Training
    trainer.fit()

if __name__ == '__main__':
    main()
