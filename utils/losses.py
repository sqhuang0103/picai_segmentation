import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation, following PI-CAI baseline."""

    def __init__(self, alpha=1.0, gamma=1.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, 2, D, H, W), targets: (B, D, H, W) long
        probs = F.softmax(inputs, dim=1)  # (B, 2, D, H, W)
        targets_onehot = F.one_hot(targets, num_classes=2).permute(0, 4, 1, 2, 3).float()  # (B, 2, D, H, W)

        p_t = (probs * targets_onehot).sum(dim=1)  # (B, D, H, W)
        ce = -torch.log(p_t + 1e-8)  # (B, D, H, W)
        focal_weight = (1 - p_t) ** self.gamma  # (B, D, H, W)

        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())  # (B, D, H, W)
        loss = alpha_t * focal_weight * ce  # (B, D, H, W)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, 2, D, H, W), targets: (B, D, H, W) long
        probs = F.softmax(inputs, dim=1)[:, 1]  # (B, D, H, W)
        targets_f = targets.float()  # (B, D, H, W)

        intersection = (probs * targets_f).sum(dim=(1, 2, 3))  # (B,)
        union = probs.sum(dim=(1, 2, 3)) + targets_f.sum(dim=(1, 2, 3))  # (B,)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B,)
        return 1.0 - dice.mean()


class FocalDiceLoss(nn.Module):
    """Focal + Dice combination loss."""

    def __init__(self, alpha=1.0, gamma=1.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + \
               self.dice_weight * self.dice(inputs, targets)
