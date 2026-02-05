import os
import contextlib
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from data.dataset import FaceEmoreDataset, BinPairDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.model_factory import build_model
from utils.logger import setup_logger
from utils.checkpoint import CheckpointManager
from engine.evaluator import Evaluator

class Trainer:
    def __init__(self, config):
        self.config = config
        self.train_cfg = config['training']
        self.data_cfg = config['data']
        self.eval_cfg = config.get('eval', {})
        
        # Setup Logger
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging_cfg = config.get('logging', {})
        log_dir = logging_cfg.get('log_dir', 'logs')
        self.logger, self.tb_writer = setup_logger(
            log_dir,
            use_tensorboard=logging_cfg.get('tensorboard', True),
            timestamp=self.timestamp,
            log_type='train'
        )
        self.logger.info(f"Run timestamp: {self.timestamp}")
        
        # Log Configuration
        import yaml
        self.logger.info("Configuration:\n" + yaml.dump(config, sort_keys=False))
        
        # Setup Device
        device_req = str(config.get('device', 'cuda')).lower()
        use_cuda = torch.cuda.is_available() and device_req != 'cpu'
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        self.use_amp = bool(self.train_cfg.get('amp', True) and self.device.type == 'cuda')
        
        # Build Model
        self.logger.info("Building model...")
        self.model = build_model(config).to(self.device)
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1 and self.train_cfg.get('data_parallel', True):
            self.model = torch.nn.DataParallel(self.model)
            
        # Optimizer & Scheduler
        optimizer_name = str(self.train_cfg.get('optimizer', 'sgd')).lower()
        if optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.train_cfg['lr'],
                momentum=self.train_cfg.get('momentum', 0.9),
                weight_decay=self.train_cfg.get('weight_decay', 0.0),
                nesterov=True
            )
        elif optimizer_name in {'adamw', 'adam'}:
            opt_cls = optim.AdamW if optimizer_name == 'adamw' else optim.Adam
            self.optimizer = opt_cls(
                self.model.parameters(),
                lr=self.train_cfg['lr'],
                weight_decay=self.train_cfg.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        if self.train_cfg.get('scheduler', 'multistep') == 'cosine':
            # CosineAnnealingLR
            # T_max is typically the total number of epochs, but we should subtract warmup epochs
            # or handle it carefully. Here we use epochs - warmup.
            # But simpler is to use epochs and let warmup override it initially.
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_cfg['epochs'],
                eta_min=self.train_cfg.get('min_lr', 1e-5)
            )
        else:
            # Default to MultiStepLR
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, 
                milestones=self.train_cfg['milestones'], 
                gamma=self.train_cfg['gamma']
            )
        
        # Checkpoint Manager
        checkpoint_cfg = config.get('checkpoint', {})
        save_dir = checkpoint_cfg.get('save_dir', 'checkpoints')
        self.save_dir = os.path.join(save_dir, self.timestamp)
        self.checkpoint_mgr = CheckpointManager(self.save_dir)
        self.logger.info(f"Checkpoints will be saved to: {self.save_dir}")
        
        # Evaluator
        self.evaluator = Evaluator(self.device)
        
        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_acc = 0.0
        self.patience_counter = 0

    def load_data(self):
        self.logger.info("Loading datasets...")
        
        # Training Data
        train_transform = get_train_transforms(self.data_cfg['img_size'])
        train_dataset = FaceEmoreDataset(
            root_dir=self.data_cfg['root'],
            transform=train_transform
        )
        pin_memory = self.device.type == 'cuda'
        num_workers = int(self.data_cfg.get('num_workers', 0))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_cfg.get('batch_size', 256),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=True
        )
        
        # Validation Data
        val_transform = get_val_transforms(self.data_cfg['img_size'])
        eval_dataset = BinPairDataset(
            root_dir=self.data_cfg['root'],
            bin_name=self.eval_cfg.get('bin_file', 'lfw.bin'),
            transform=val_transform
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.eval_cfg.get('test_batch_size', 256),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        self.logger.info(f"Train samples: {len(train_dataset):,}, Classes: {train_dataset.num_classes:,}")
        self.logger.info(f"Evaluation pairs: {len(eval_dataset):,}")
        expected_classes = self.config.get('head', {}).get('num_classes', None)
        if expected_classes is not None and int(expected_classes) != int(train_dataset.num_classes):
            self.logger.warning(f"head.num_classes={expected_classes} != dataset.num_classes={train_dataset.num_classes}")

    def resume(self):
        if not self.train_cfg.get('resume', False):
            return

        latest_ckpt = None
        base_dir = self.config.get('checkpoint', {}).get('save_dir', None)
        if not base_dir:
            self.logger.warning("Resume requested but checkpoint.save_dir is not set")
            return
        if os.path.exists(base_dir):
            try:
                dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                dirs.sort(key=os.path.getmtime)
                if dirs:
                    latest_dir = dirs[-1]
                    ckpt_path = os.path.join(latest_dir, 'last.pth')
                    if os.path.exists(ckpt_path):
                        latest_ckpt = ckpt_path
            except Exception as e:
                self.logger.warning(f"Error finding checkpoint: {e}")
        
        if latest_ckpt:
            self.logger.info(f"Resuming from: {latest_ckpt}")
            try:
                loaded_epoch, loaded_patience = self.checkpoint_mgr.load(
                    self.model, latest_ckpt, self.optimizer, self.scheduler, self.device
                )
                self.start_epoch = loaded_epoch
                self.patience_counter = loaded_patience
                self.best_acc = self.checkpoint_mgr.best_metric
                self.global_step = self.start_epoch * len(self.train_loader)
                self.logger.info(f"Resumed at Epoch {self.start_epoch + 1}")
            except Exception as e:
                self.logger.error(f"Resume failed: {e}")

    def train_epoch(self, epoch):
        # Update dynamic margins if needed
        if hasattr(self.model, 'module'):
            if hasattr(self.model.module.head, 'update_margin'):
                self.model.module.head.update_margin(epoch)
        elif hasattr(self.model.head, 'update_margin'):
            self.model.head.update_margin(epoch)

        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        running_acc5 = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1} [Train]')
        scaler = torch.cuda.amp.GradScaler(init_scale=float(self.train_cfg.get('amp_init_scale', 128.0))) if self.use_amp else None
        autocast_ctx = torch.cuda.amp.autocast if self.use_amp else contextlib.nullcontext
        grad_clip_norm = float(self.train_cfg.get('grad_clip_norm', 5.0))

        # Warmup parameters
        warmup_epochs = self.train_cfg.get('warmup_epochs', 0)
        base_lr = self.train_cfg['lr']
        total_steps = len(self.train_loader) * int(warmup_epochs) if warmup_epochs else 0
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Per-iteration Warmup
            if epoch < warmup_epochs:
                current_step = epoch * len(self.train_loader) + batch_idx
                warmup_lr = base_lr * (current_step + 1) / max(total_steps, 1)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast_ctx():
                logits, _ = self.model(images, labels)
                loss = F.cross_entropy(logits, labels)
            
            if not torch.isfinite(loss).item():
                self.logger.warning(f"Non-finite loss at epoch={epoch + 1}, step={batch_idx}: {loss.item()}")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item() * 100
                _, pred5 = logits.topk(5, dim=1)
                acc5 = (pred5 == labels.view(-1, 1)).sum().float() / labels.size(0) * 100
            
            running_loss += loss.item()
            running_acc += acc
            running_acc5 += acc5.item()
            self.global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = running_acc / len(self.train_loader)
        epoch_acc5 = running_acc5 / len(self.train_loader)

        self.logger.info(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Top-5 Acc: {epoch_acc5:.2f}%")
        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss', epoch_loss, epoch + 1)
            self.tb_writer.add_scalar('train/acc', epoch_acc, epoch + 1)
            self.tb_writer.add_scalar('train/acc5', epoch_acc5, epoch + 1)
            self.tb_writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch + 1)

    def fit(self):
        self.load_data()
        self.resume()
        
        self.logger.info("Starting training...")
        for epoch in range(self.start_epoch, self.train_cfg['epochs']):
            self.train_epoch(epoch)
            
            # Validation
            if (epoch + 1) % self.eval_cfg.get('eval_freq', 1) == 0:
                metrics = self.evaluator.validate(self.model, self.eval_loader)
                val_acc = metrics['accuracy']
                
                self.logger.info(f"Validation Accuracy: {val_acc:.4f} ± {metrics['std']:.4f} (avg threshold={metrics['best_threshold']:.4f})")
                if self.tb_writer:
                    self.tb_writer.add_scalar('val/accuracy', val_acc, epoch + 1)
                    self.tb_writer.add_scalar('val/auc', metrics['auc'], epoch + 1)

                is_best = val_acc > self.best_acc
                self.checkpoint_mgr.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    metric=val_acc,
                    is_best=is_best,
                    patience=self.patience_counter
                )

                if is_best:
                    self.best_acc = val_acc
                    self.patience_counter = 0
                    self.logger.info(f"✓ New best accuracy: {self.best_acc:.4f}")
                elif epoch + 1 > 30:
                    self.patience_counter += 1
                    self.logger.info(f"No improvement for {self.patience_counter} epochs")
                    
                    # Early Stopping Check
                    if self.train_cfg.get('early_stop', False):
                        limit = self.train_cfg.get('patience', 10)
                        if self.patience_counter >= limit:
                            self.logger.info(f"Early stopping triggered (No improvement for {limit} epochs after epoch 30)")
                            break
                else:
                    self.patience_counter = 0
            
            self.scheduler.step()

        if self.tb_writer:
            self.tb_writer.close()
