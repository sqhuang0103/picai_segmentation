import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from configs.config import *
from models.unet import UNet
from data.dataset import PICAIPreprocessedDataset
from data.augmentations import PicaiAugmentation
from utils.losses import FocalLoss
from utils.metrics import compute_dice


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ids = SPLITS[str(FOLD)]["train"]
    val_ids = SPLITS[str(FOLD)]["val"]

    augmentation = PicaiAugmentation()
    train_set = PICAIPreprocessedDataset(PREPROCESSED_DIR, train_ids, transform=augmentation)
    val_set = PICAIPreprocessedDataset(PREPROCESSED_DIR, val_ids, transform=None)

    print(f"Fold {FOLD}: train={len(train_set)}, val={len(val_set)}")

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
        for images, labels in train_loader:
            images = images.float().to(device)  # (B, 3, D, H, W)
            labels = labels[:, 0].long().to(device)  # (B, D, H, W)
            logits = model(images)  # (B, 2, D, H, W)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= max(len(train_set), 1)

        if epoch % VALIDATE_N_EPOCHS == 0 or epoch == NUM_EPOCHS:
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.float().to(device)  # (B, 3, D, H, W)
                    labels_np = labels[:, 0].numpy()  # (B, D, H, W)
                    logits = model(images)  # (B, 2, D, H, W)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B, D, H, W)
                    for i in range(preds.shape[0]):
                        val_dices.append(compute_dice(preds[i], labels_np[i]))
            mean_dice = np.mean(val_dices) if val_dices else 0.0

            print(f"Epoch {epoch}/{NUM_EPOCHS}  loss={train_loss:.4f}  val_dice={mean_dice:.4f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
                print(f"  -> saved best model (dice={best_dice:.4f})")
        else:
            print(f"Epoch {epoch}/{NUM_EPOCHS}  loss={train_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    print(f"Training done. Best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    train()
