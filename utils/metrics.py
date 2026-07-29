import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_dice(pred, target, smooth=1e-5):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def compute_dice_strict(pred, target):
    """Dice without smoothing — returns NaN when both pred and target are empty."""
    pred_flat = pred.flatten().astype(float)
    target_flat = target.flatten().astype(float)
    intersection = (pred_flat * target_flat).sum()
    denom = pred_flat.sum() + target_flat.sum()
    if denom == 0:
        return float("nan")
    return 2.0 * intersection / denom


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
        probs: (N, D, H, W) voxel-level probability maps
        preds: (N, D, H, W) binary segmentation predictions
        targets: (N, D, H, W) binary segmentation ground truth
    Returns:
        dict with AUC, Dice, Sen@90Spe, Spe@90Sen (all lesion-level)
    """
    dices_all = []
    dices_pos = []
    for i in range(len(preds)):
        dices_all.append(compute_dice(preds[i], targets[i]))
        d = compute_dice_strict(preds[i], targets[i])
        if not np.isnan(d):
            dices_pos.append(d)

    # lesion-level: pool voxels from positive samples only
    voxel_probs, voxel_labels = [], []
    for i in range(len(targets)):
        if targets[i].sum() > 0:
            voxel_probs.append(probs[i].flatten())
            voxel_labels.append(targets[i].flatten())
    if voxel_probs:
        voxel_probs = np.concatenate(voxel_probs)
        voxel_labels = np.concatenate(voxel_labels).astype(float)
    else:
        voxel_probs = np.array([])
        voxel_labels = np.array([])

    return {
        "AUC": compute_auc(voxel_probs, voxel_labels),
        "Dice": np.mean(dices_all),
        "Dice_lesion": np.mean(dices_pos) if dices_pos else 0.0,
        "Sen@90Spe": sensitivity_at_specificity(voxel_probs, voxel_labels, 0.90),
        "Spe@90Sen": specificity_at_sensitivity(voxel_probs, voxel_labels, 0.90),
    }
