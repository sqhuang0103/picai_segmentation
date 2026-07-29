import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.config import *
from models.unet import UNet
from data.dataset import PICAIPreprocessedDataset
from data.augmentations import PicaiAugmentation
from utils.losses import FocalDiceLoss
from utils.metrics import compute_dice_strict


def _build_sampler(dataset):
    """Oversample lesion-positive cases to ~50% of each epoch."""
    weights = []
    n_pos, n_neg = 0, 0
    for sid in dataset.samples:
        label_path = os.path.join(dataset.labels_dir, f"{sid}.nii.gz")
        import SimpleITK as sitk
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        if label.sum() > 0:
            weights.append(1.0)
            n_pos += 1
        else:
            weights.append(0.0)
            n_neg += 1
    w_pos = 1.0 / max(n_pos, 1)
    w_neg = 1.0 / max(n_neg, 1)
    weights = [w_pos if w > 0 else w_neg for w in weights]
    print(f"Sampler: {n_pos} positive, {n_neg} negative cases")
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids = SPLITS[str(FOLD)]["train"]
    val_ids = SPLITS[str(FOLD)]["val"]

    augmentation = PicaiAugmentation()
    train_set = PICAIPreprocessedDataset(PREPROCESSED_DIR, train_ids, transform=augmentation)
    val_set = PICAIPreprocessedDataset(PREPROCESSED_DIR, val_ids, transform=None)

    print(f"Fold {FOLD}: train={len(train_set)}, val={len(val_set)}")

    sampler = _build_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)

    alpha = 1.0 - POSITIVE_RATIO
    criterion = FocalDiceLoss(alpha=alpha, gamma=FOCAL_LOSS_GAMMA,
                              dice_weight=1.0, focal_weight=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_dice = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.float().to(device)
            labels = labels[:, 0].long().to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= max(len(train_set), 1)
        scheduler.step()

        if epoch % VALIDATE_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.float().to(device)
                    labels_np = labels[:, 0].numpy()
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    for i in range(preds.shape[0]):
                        d = compute_dice_strict(preds[i], labels_np[i])
                        if not np.isnan(d):
                            val_dices.append(d)
            mean_dice = np.mean(val_dices) if val_dices else 0.0

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{NUM_EPOCHS}  loss={train_loss:.4f}  "
                  f"val_dice_lesion={mean_dice:.4f}  lr={lr_now:.2e}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_v2.pth"))
                print(f"  -> saved best model (dice_lesion={best_dice:.4f})")
        else:
            print(f"Epoch {epoch}/{NUM_EPOCHS}  loss={train_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model_v2.pth"))
    print(f"Training done. Best val dice (lesion-only): {best_dice:.4f}")


if __name__ == "__main__":
    train()
