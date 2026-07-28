"""
Offline preprocessing: raw .mha -> resample + center crop/pad + z-score -> .nii.gz

Output structure:
  preprocessed/
    images/
      {subject_id}.nii.gz     # (C, D, H, W) = (3, 20, 256, 256) float32
    labels/
      {subject_id}.nii.gz     # (D, H, W) = (20, 256, 256) uint8
"""

import os
import sys
import json
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.dataset import resample_volume, center_crop_or_pad, zscore_normalize

TARGET_SPACING = (3.0, 0.5, 0.5)
TARGET_SIZE = (20, 256, 256)
MODALITIES = ["t2w", "adc", "hbv"]

DATA_ROOT = os.environ.get("PICAI_DATA_ROOT", "/home/rmapshu/Scratch/picai_data")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
ANNOTATION_DIR = os.path.join(DATA_ROOT, "picai_labels",
                              "csPCa_lesion_delineations", "human_expert", "resampled")
OUTPUT_DIR = os.path.join(DATA_ROOT, "preprocessed")
OUT_IMAGES = os.path.join(OUTPUT_DIR, "images")
OUT_LABELS = os.path.join(OUTPUT_DIR, "labels")

SPLITS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "splits.json")


def process_subject(subject_id):
    patient_id = subject_id.split("_")[0]
    patient_dir = os.path.join(IMAGES_DIR, patient_id)

    out_img_path = os.path.join(OUT_IMAGES, f"{subject_id}.nii.gz")
    out_lbl_path = os.path.join(OUT_LABELS, f"{subject_id}.nii.gz")

    if os.path.exists(out_img_path) and os.path.exists(out_lbl_path):
        return f"[skip] {subject_id}"

    channels = []
    for mod in MODALITIES:
        path = os.path.join(patient_dir, f"{subject_id}_{mod}.mha")
        if not os.path.exists(path):
            return f"[miss] {subject_id}: {mod} not found"
        img = sitk.ReadImage(path)
        img = resample_volume(img, TARGET_SPACING, is_label=False)
        vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (D, H, W)
        vol = center_crop_or_pad(vol, TARGET_SIZE)  # (20, 256, 256)
        vol = zscore_normalize(vol)  # (20, 256, 256)
        channels.append(vol)

    image = np.stack(channels, axis=0)  # (3, 20, 256, 256)
    img_sitk = sitk.GetImageFromArray(image)  # sitk treats first dim as depth
    sitk.WriteImage(img_sitk, out_img_path)

    label_path = os.path.join(ANNOTATION_DIR, f"{subject_id}.nii.gz")
    if os.path.exists(label_path):
        lbl = sitk.ReadImage(label_path)
        lbl = resample_volume(lbl, TARGET_SPACING, is_label=True)
        lbl_arr = sitk.GetArrayFromImage(lbl).astype(np.float32)  # (D, H, W)
        lbl_arr = center_crop_or_pad(lbl_arr, TARGET_SIZE)  # (20, 256, 256)
        lbl_arr = (lbl_arr > 0).astype(np.uint8)  # (20, 256, 256)
    else:
        lbl_arr = np.zeros(TARGET_SIZE, dtype=np.uint8)  # (20, 256, 256)

    lbl_sitk = sitk.GetImageFromArray(lbl_arr)
    sitk.WriteImage(lbl_sitk, out_lbl_path)

    return f"[done] {subject_id}"


def main():
    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_LABELS, exist_ok=True)

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    all_ids = set()
    for fold in splits.values():
        all_ids.update(fold["train"])
        all_ids.update(fold["val"])
    all_ids = sorted(all_ids)
    print(f"Total subjects to preprocess: {len(all_ids)}")

    n_workers = int(os.environ.get("NSLOTS", 4))
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_subject, all_ids)):
            if (i + 1) % 50 == 0 or (i + 1) == len(all_ids):
                print(f"  [{i+1}/{len(all_ids)}] {result}")

    print("Preprocessing done.")


if __name__ == "__main__":
    main()
