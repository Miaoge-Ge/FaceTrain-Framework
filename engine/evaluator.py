import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.metrics import evaluate_lfw_10fold, calculate_tar_at_far

class Evaluator:
    def __init__(self, device):
        self.device = device

    def validate(self, model, loader):
        model.eval()
        similarities = []
        labels_list = []

        # print("Evaluating...") # Can be controlled by logger in caller
        with torch.no_grad():
            for img1, img2, label in tqdm(loader, desc="Validation", leave=False):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                feat1 = model(img1, labels=None)
                feat2 = model(img2, labels=None)
                
                feat1 = F.normalize(feat1, p=2, dim=1)
                feat2 = F.normalize(feat2, p=2, dim=1)
                
                sim = (feat1 * feat2).sum(dim=1).cpu().numpy()
                similarities.extend(sim)
                labels_list.extend(label.numpy() if isinstance(label, torch.Tensor) else label)

        similarities = np.array(similarities)
        labels = np.array(labels_list, dtype=bool)

        # Use 10-fold Cross Validation for Accuracy
        val_acc, val_std, best_thr = evaluate_lfw_10fold(similarities, labels)
        
        # Calculate TAR @ FAR
        far_levels = [1e-2, 1e-3]
        tar_at_far, roc_auc = calculate_tar_at_far(similarities, labels, far_levels=far_levels)
        
        return {
            'accuracy': val_acc,
            'std': val_std,
            'best_threshold': best_thr,
            'tar_at_far': tar_at_far,
            'auc': roc_auc
        }
