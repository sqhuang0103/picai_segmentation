import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_dice(pred, target, smooth=1e-5):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def compute_auc(probs, targets):
    if len(np.unique(targets)) < 2:
        return float("nan")
    return roc_auc_score(targets, probs)


def sensitivity_at_specificity(probs, targets, target_spec=0.90):
    if len(np.unique(targets)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(targets, probs)
    spec = 1.0 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return 0.0
    return tpr[idx[-1]]


def specificity_at_sensitivity(probs, targets, target_sen=0.90):
    if len(np.unique(targets)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(targets, probs)
    idx = np.where(tpr >= target_sen)[0]
    if len(idx) == 0:
        return 0.0
    return 1.0 - fpr[idx[0]]


def evaluate_metrics(probs, preds, targets):
    """
    Args:
        probs: (N,) patient-level prediction probabilities (max over volume)
        preds: (N, D, H, W) binary segmentation predictions
        targets: (N, D, H, W) binary segmentation ground truth
    Returns:
        dict with AUC, Dice, Sen@90Spe, Spe@90Sen
    """
    patient_targets = (targets.reshape(targets.shape[0], -1).sum(axis=1) > 0).astype(float)

    dices = []
    for i in range(len(preds)):
        dices.append(compute_dice(preds[i], targets[i]))

    return {
        "AUC": compute_auc(probs, patient_targets),
        "Dice": np.mean(dices),
        "Sen@90Spe": sensitivity_at_specificity(probs, patient_targets, 0.90),
        "Spe@90Sen": specificity_at_sensitivity(probs, patient_targets, 0.90),
    }
