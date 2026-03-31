import os
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

def get_config(debug=False):
    """
    Returns datasets, seeds, and percentages for testing.
    If debug=True, returns a very small subset for fast regression testing.
    """
    if debug:
        datasets = ['algo004']
        seeds = [18]
        percentages = [0.1, 0.25]
    else:
        datasets = ['algo004', 'comp', 'ml', 'virtualshakespeare']
        seeds = [18, 61, 53, 29, 69, 42, 2, 21, 78, 99]
        percentages = [0.1, 0.25, 0.5, 0.75]
    return datasets, seeds, percentages

def compute_loss_gnn(pos_score, neg_score, device, max_weight=20):
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([
        torch.ones(pos_score.shape[0], device=scores.device), 
        torch.zeros(neg_score.shape[0], device=scores.device)
    ])

    num_pos = pos_score.shape[0]
    num_neg = neg_score.shape[0]
    # Handle division by zero
    if num_pos > 0:
        pos_weight = torch.tensor([num_neg / num_pos], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)
        
    pos_weight = torch.clamp(pos_weight, max=max_weight)
    return F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)

def compute_auc_gnn(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().detach().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().detach().numpy()
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)

def compute_acc_gnn(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().detach().numpy()
    preds = (torch.sigmoid(torch.tensor(scores)) >= 0.5).numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().detach().numpy()
    return np.mean(preds == labels)

def balanced_error_rate(cm):
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return 0.5 * (fpr + fnr)

def calculate_metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    total_samples = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    tp_div_all = tp / total_samples if total_samples > 0 else 0
    fp_div_all = fp / total_samples if total_samples > 0 else 0
    return precision, tp_div_all, fp_div_all
