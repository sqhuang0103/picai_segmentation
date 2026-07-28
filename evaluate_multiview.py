"""
Evaluate multi-view trained model using deterministic center crop.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from configs.config import *
from models.unet import UNet
from data.dataset import PICAIMultiViewDataset
from utils.metrics import evaluate_metrics

CROP_SIZE = (16, 192, 192)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ids = SPLITS[str(FOLD)]["val"]
    val_set = PICAIMultiViewDataset(PREPROCESSED_DIR, val_ids, crop_size=CROP_SIZE,
                                    num_views=1, is_train=False, transform=None)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    ckpt_path = os.path.join(OUTPUT_DIR, "best_multiview.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.float().to(device)  # (B, C, D', H', W')
            logits = model(images)  # (B, 2, D', H', W')
            prob_map = torch.softmax(logits, dim=1)[:, 1]  # (B, D', H', W')
            preds = (prob_map > 0.5).cpu().numpy().astype(float)  # (B, D', H', W')
            probs = prob_map.cpu().numpy()  # (B, D', H', W')

            patient_probs = probs.reshape(probs.shape[0], -1).max(axis=1)  # (B,)
            all_probs.append(patient_probs)
            all_preds.append(preds)
            all_targets.append(labels[:, 0].numpy())  # (B, D', H', W')

    all_probs = np.concatenate(all_probs)  # (N,)
    all_preds = np.concatenate(all_preds)  # (N, D', H', W')
    all_targets = np.concatenate(all_targets)  # (N, D', H', W')

    results = evaluate_metrics(all_probs, all_preds, all_targets)

    print("=" * 40)
    print(f"MultiView Evaluation Results (Fold {FOLD})")
    print("=" * 40)
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate()
