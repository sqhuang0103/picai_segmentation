"""
Multi-view training: K random crops per volume, each with its own label.

L_total = (1/K) * sum(L_focal(logits_k, label_k))
        + lambda * consistency_loss

Consistency regularisation penalises variance of per-view mean predicted
probability (patient-level), encouraging calibrated predictions regardless
of crop location.

lambda is linearly warmed up from 0 over WARMUP_EPOCHS.
At inference, center crop is used.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from configs.config import *
from models.unet import UNet
from data.dataset import PICAIMultiViewDataset
from data.augmentations import PicaiAugmentation
from utils.losses import FocalLoss
from utils.metrics import compute_dice

# Multi-view hyperparameters
NUM_VIEWS = 4
CROP_SIZE = (16, 192, 192)
LAMBDA_CONSISTENCY = 1.0
WARMUP_EPOCHS = 10


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids = SPLITS[str(FOLD)]["train"]
    val_ids = SPLITS[str(FOLD)]["val"]

    augmentation = PicaiAugmentation()
    train_set = PICAIMultiViewDataset(PREPROCESSED_DIR, train_ids, crop_size=CROP_SIZE,
                                      num_views=NUM_VIEWS, is_train=True, transform=augmentation)
    val_set = PICAIMultiViewDataset(PREPROCESSED_DIR, val_ids, crop_size=CROP_SIZE,
                                    num_views=1, is_train=False, transform=None)

    print(f"[MultiView] Fold {FOLD}: train={len(train_set)}, val={len(val_set)}, "
          f"K={NUM_VIEWS}, crop={CROP_SIZE}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)

    alpha = 1.0 - POSITIVE_RATIO
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_LOSS_GAMMA).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    best_dice = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        # linear warmup for consistency weight
        if epoch <= WARMUP_EPOCHS:
            lam = LAMBDA_CONSISTENCY * (epoch - 1) / WARMUP_EPOCHS
        else:
            lam = LAMBDA_CONSISTENCY

        for images, labels in train_loader:
            # images: (B, K, C, D', H', W'), labels: (B, K, D', H', W')
            B, K = images.shape[0], images.shape[1]

            # reshape to process all views in one forward pass
            images_flat = images.reshape(B * K, *images.shape[2:]).float().to(device)  # (B*K, C, D', H', W')
            labels_flat = labels.reshape(B * K, *labels.shape[2:]).long().to(device)   # (B*K, D', H', W')

            logits_flat = model(images_flat)  # (B*K, 2, D', H', W')

            # per-view focal loss: each view uses its own spatial label
            loss_main = criterion(logits_flat, labels_flat)  # averaged over B*K views

            # consistency regularisation: per-view mean probability should be similar
            logits_views = logits_flat.reshape(B, K, *logits_flat.shape[1:])  # (B, K, 2, D', H', W')
            probs_views = torch.softmax(logits_views, dim=2)[:, :, 1]  # (B, K, D', H', W')
            mean_prob_per_view = probs_views.mean(dim=(2, 3, 4))  # (B, K)
            mean_prob = mean_prob_per_view.mean(dim=1, keepdim=True).detach()  # (B, 1)
            loss_consistency = ((mean_prob_per_view - mean_prob) ** 2).mean()

            loss = loss_main + lam * loss_consistency

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * B
        train_loss /= max(len(train_set), 1)

        # --- Validate (center crop, single view) ---
        if epoch % VALIDATE_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.float().to(device)  # (B, C, D', H', W')
                    labels_np = labels[:, 0].numpy()  # (B, D', H', W')
                    logits = model(images)  # (B, 2, D', H', W')
                    preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B, D', H', W')
                    for i in range(preds.shape[0]):
                        val_dices.append(compute_dice(preds[i], labels_np[i]))
            mean_dice = np.mean(val_dices) if val_dices else 0.0

            print(f"Epoch {epoch}/{NUM_EPOCHS}  loss={train_loss:.4f}  "
                  f"val_dice={mean_dice:.4f}  lam={lam:.3f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_multiview.pth"))
                print(f"  -> saved best model (dice={best_dice:.4f})")
        else:
            print(f"Epoch {epoch}/{NUM_EPOCHS}  loss={train_loss:.4f}  lam={lam:.3f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_multiview.pth"))
    print(f"Training done. Best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    train()
