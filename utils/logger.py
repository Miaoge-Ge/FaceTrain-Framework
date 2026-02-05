import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def setup_logger(log_dir, use_tensorboard=True, timestamp=None, log_type='train', name=None):
    """
    Setup unified logger.
    Args:
        log_dir: Directory to save logs (e.g., 'logs')
        use_tensorboard: Whether to use tensorboard
        timestamp: Timestamp string
        log_type: 'train' or 'test' (prefix for log file)
        name: Optional name suffix (e.g., test run name)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Construct log filename: type_name_timestamp.log or type_timestamp.log
    if name:
        filename = f'{log_type}_{name}_{timestamp}.log'
    else:
        filename = f'{log_type}_{timestamp}.log'
        
    log_file = os.path.join(log_dir, filename)
    
    # Use a unique logger name to avoid conflicts if multiple loggers are created
    logger_name = f'FaceRecognition_{log_type}_{timestamp}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    tb_writer = None
    if use_tensorboard and log_type == 'train':
        tb_dir = os.path.join(log_dir, 'tensorboard', timestamp)
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard logs saved to: {tb_dir}")
    
    logger.info(f"Logging started. Log file: {log_file}")
    
    return logger, tb_writer
