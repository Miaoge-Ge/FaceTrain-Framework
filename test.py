import os
import yaml
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import logging
import argparse

from models.model_factory import build_model
from data.dataset import BinPairDataset
from data.transforms import get_val_transforms
from engine.evaluator import Evaluator

from utils.logger import setup_logger

def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        import numpy
        torch.serialization.add_safe_globals([
            numpy.core.multiarray.scalar, 
            numpy.dtype,
            numpy.float64,
            numpy.int64
        ])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded model trained for {epoch} epochs")
    return epoch

def run_test(config_path, checkpoint_path, save_dir=None, name=None, use_cpu=False):
    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Determine log directory
    # Priority: 
    # 1. Function argument save_dir
    # 2. Config 'logging.log_dir'
    # 3. Default 'logs'
    if save_dir is None:
        save_dir = config.get('logging', {}).get('log_dir', 'logs')
    
    # Generate timestamp and run ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = name if name else f"test_{timestamp}"

    # Setup save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup Logger
    logger, _ = setup_logger(
        save_dir, 
        use_tensorboard=False, 
        timestamp=timestamp, 
        log_type='test', 
        name=name
    )
    
    logger.info(f"Starting evaluation run: {run_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare Data
    data_cfg = config['data']
    eval_cfg = config.get('eval', {})
    if not os.path.exists(data_cfg['root']):
        raise FileNotFoundError(f"Dataset root not found: {data_cfg['root']}")
    
    img_size = data_cfg['img_size']
    if isinstance(img_size, list):
        img_size = tuple(img_size)
    val_transform = get_val_transforms(img_size)
    
    eval_dataset = BinPairDataset(
        root_dir=data_cfg['root'],
        bin_name=eval_cfg.get('bin_file', 'lfw.bin'),
        transform=val_transform
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_cfg.get('test_batch_size', 256),
        shuffle=False,
        num_workers=int(data_cfg.get('num_workers', 0)),
        pin_memory=(device.type == 'cuda'),
        persistent_workers=int(data_cfg.get('num_workers', 0)) > 0
    )
    
    # Build Model
    model = build_model(config).to(device)
    
    # Load Checkpoint
    trained_epochs = load_checkpoint(model, checkpoint_path, device)
    
    # Run Evaluation using Engine
    evaluator = Evaluator(device)
    metrics = evaluator.validate(model, eval_loader)
    
    # Log Results
    logger.info("-" * 50)
    logger.info(f"Test Results for {run_name}")
    logger.info("-" * 50)
    logger.info(f"Checkpoint: {checkpoint_path} (Epoch {trained_epochs})")
    logger.info(f"Validation Dataset: {eval_cfg.get('dataset', 'Unknown')}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f} ± {metrics['std']:.4f} (avg threshold={metrics['best_threshold']:.4f})")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info("-" * 50)
    logger.info("TAR @ FAR:")
    for far, (tar, threshold) in metrics['tar_at_far'].items():
        logger.info(f"FAR={far:.1e}: TAR={tar:.4f} (Threshold={threshold:.4f})")
    logger.info("-" * 50)
    
    # Save results to a result file in the same logs directory, but maybe we want a specific result file?
    # The user asked for "unified logs directory".
    # We can save a summary file as well, or just rely on the log file.
    # Let's save a summary file named 'test_result_TIMESTAMP.txt' in the same log dir.
    
    result_filename = f'test_result_{timestamp}.txt'
    if name:
        result_filename = f'test_result_{name}_{timestamp}.txt'
        
    result_path = os.path.join(save_dir, result_filename)
    
    with open(result_path, 'w') as f:
        f.write(f"Run Name: {run_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Validation Dataset: {eval_cfg.get('dataset', 'Unknown')}\n")
        f.write(f"Trained Epochs: {trained_epochs}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ± {metrics['std']:.4f} (avg threshold={metrics['best_threshold']:.4f})\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write("-" * 20 + "\n")
        f.write("TAR @ FAR:\n")
        for far, (tar, threshold) in metrics['tar_at_far'].items():
            f.write(f"FAR={far:.1e}: TAR={tar:.4f} (Threshold={threshold:.4f})\n")
    
    logger.info(f"Summary results saved to {result_path}")
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Face Recognition Evaluation")
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint (.pth)')
    parser.add_argument('--save-dir', type=str, default=None, help='directory to save logs/results')
    parser.add_argument('--name', type=str, default=None, help='run name suffix for logs/results')
    parser.add_argument('--cpu', action='store_true', help='force CPU')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_test(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        name=args.name,
        use_cpu=args.cpu
    )
