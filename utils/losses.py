import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    logits: 模型原始输出 (N, 1, H, W)
    targets: 0/1 mask (N, 1, H, W)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        return loss.mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smooth=1, dice_weight=0.5, focal_weight=0.5):
        super(DiceFocalLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        # ---- Dice Loss ----
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / \
                    (probs.sum() + targets.sum() + self.smooth)
        focal_loss = self.focal(logits, targets.view(-1, 1, logits.shape[2], logits.shape[3]))
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


def DiceBCELoss(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

