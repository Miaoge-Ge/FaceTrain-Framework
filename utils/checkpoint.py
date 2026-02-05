import os
import torch
import logging

class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_path = os.path.join(save_dir, 'best.pth')
        self.last_path = os.path.join(save_dir, 'last.pth')
        self.best_metric = 0.0
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, optimizer, scheduler, epoch, metric=None, is_best=False, patience=0):
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        # 将 numpy 类型转换为 Python 原生类型，以符合 weights_only=True 的最佳实践
        if metric is not None and hasattr(metric, 'item'):
            metric = metric.item()
        
        state = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metric': metric,
            'patience': patience,
        }
        
        # 始终保存/覆盖 last.pth
        torch.save(state, self.last_path)
        self.logger.info(f"Checkpoint saved: {self.last_path}")
        
        if is_best or (metric is not None and metric > self.best_metric):
            self.best_metric = metric or self.best_metric
            torch.save(state, self.best_path)
            self.logger.info(f"New best model saved: {self.best_path} (metric: {self.best_metric:.4f})")
    
    def load(self, model, checkpoint_path, optimizer=None, scheduler=None, device='cpu'):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # 尝试标准加载，如果失败则启用兼容模式
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as e:
            self.logger.warning(f"标准加载失败，尝试添加安全白名单: {e}")
            import numpy
            torch.serialization.add_safe_globals([
                numpy.core.multiarray.scalar, 
                numpy.dtype,
                numpy.float64,
                numpy.int64
            ])
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        state_dict = checkpoint['model_state_dict']
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metric = checkpoint.get('metric', None)
        patience = checkpoint.get('patience', 0)
        self.logger.info(f"Loaded checkpoint from epoch {epoch} (metric: {metric})")
        
        if metric is not None:
            self.best_metric = max(self.best_metric, metric)
        
        return epoch, patience
