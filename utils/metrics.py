import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

def accuracy(output, target, topk=(1,)):
    """计算top-k准确率（返回top-1准确率或tuple）"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res[0] if len(topk) == 1 else tuple(res)


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def evaluate_lfw_10fold(distances, labels, n_folds=10):
    """
    Standard LFW 10-fold validation protocol.
    For each fold, use the other 9 folds to find the best threshold, 
    then calculate accuracy on the current fold.
    """
    thresholds = np.arange(-1.0, 1.0, 0.005)
    n_pairs = len(distances)
    fold_size = n_pairs // n_folds
    
    accuracies = []
    best_thresholds = []
    
    indices = np.arange(n_pairs)
    
    for fold in range(n_folds):
        test_mask = (indices >= fold * fold_size) & (indices < (fold + 1) * fold_size)
        train_mask = ~test_mask
        
        # Train set (9 folds) to find best threshold
        dist_train = distances[train_mask]
        labels_train = labels[train_mask]
        
        best_acc_train = 0.0
        best_thresh_train = 0.0
        
        # Simple grid search for threshold
        for th in thresholds:
            _, _, acc = calculate_accuracy(th, dist_train, labels_train)
            if acc > best_acc_train:
                best_acc_train = acc
                best_thresh_train = th
        
        best_thresholds.append(best_thresh_train)
        
        # Test set (1 fold)
        dist_test = distances[test_mask]
        labels_test = labels[test_mask]
        _, _, acc_test = calculate_accuracy(best_thresh_train, dist_test, labels_test)
        accuracies.append(acc_test)
        
    return np.mean(accuracies), np.std(accuracies), np.mean(best_thresholds)

def calculate_tar_at_far(distances, labels, far_levels=[1e-3, 1e-4, 1e-5]):
    """
    Calculate True Acceptance Rate (TAR) at specific False Acceptance Rate (FAR) levels.
    """
    # sklearn roc_curve returns fpr, tpr, thresholds
    # Note: distances are cosine similarities (-1 to 1), higher is better.
    # sklearn expects higher values to indicate positive class (Same Person).
    fpr, tpr, thresholds = roc_curve(labels, distances)
    
    # Sort by FPR just in case (though roc_curve usually returns sorted)
    # But roc_curve returns decreasing thresholds, increasing FPR
    # We want to interpolate TAR for specific FAR
    
    metrics = {}
    
    # Ensure sorted by FPR for interpolation
    # roc_curve returns increasing FPR
    
    for far_target in far_levels:
        # Find index where FPR is closest to far_target but <= far_target
        # Using interpolation for smoother results
        
        # We need to find the TAR at this FAR
        tar = np.interp(far_target, fpr, tpr)
        
        # Also find the threshold at this FAR
        # Since FPR increases as threshold decreases, we interp threshold vs FPR
        # But thresholds array in roc_curve is decreasing.
        # Let's flip them for interpolation
        threshold_at_far = np.interp(far_target, fpr, thresholds)
        
        metrics[far_target] = (tar, threshold_at_far)
        
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    
    return metrics, roc_auc
