import os
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

# PI-CAI baseline preprocessing parameters
TARGET_SPACING = (3.0, 0.5, 0.5)  # mm, (D, H, W)
TARGET_SIZE = (20, 256, 256)
CLIP_LOW_PERCENTILE = 0.5
CLIP_HIGH_PERCENTILE = 99.5


def resample_volume(image, target_spacing, is_label=False):
    """Resample a SimpleITK image to target spacing."""
    original_spacing = np.array(image.GetSpacing())       # (W, H, D) in sitk
    original_size = np.array(image.GetSize())
    target_spacing_sitk = np.array(target_spacing)[::-1]  # (D,H,W) -> (W,H,D)

    new_size = np.round(original_size * original_spacing / target_spacing_sitk).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_sitk.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)


def center_crop_or_pad(vol, target_shape):
    """Center crop or zero-pad a 3D numpy array to target_shape."""
    result = np.zeros(target_shape, dtype=vol.dtype)
    slices_src, slices_dst = [], []
    for i in range(len(target_shape)):
        s, t = vol.shape[i], target_shape[i]
        if s >= t:
            start = (s - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(0, t))
        else:
            start = (t - s) // 2
            slices_src.append(slice(0, s))
            slices_dst.append(slice(start, start + s))
    result[tuple(slices_dst)] = vol[tuple(slices_src)]
    return result


def zscore_normalize(vol, low_pct=CLIP_LOW_PERCENTILE, high_pct=CLIP_HIGH_PERCENTILE):
    """Percentile clipping (0.5/99.5) + instance-wise z-score normalization."""
    low = np.percentile(vol, low_pct)
    high = np.percentile(vol, high_pct)
    vol = np.clip(vol, low, high)
    mean = vol.mean()
    std = vol.std() + 1e-8
    return (vol - mean) / std


class PICAIDataset(Dataset):
    """
    Args:
        subject_ids: list of "patientid_studyid" strings (from official splits.json)
    """
    def __init__(self, images_dir, annotation_dir, subject_ids, modalities=("t2w", "adc", "hbv"),
                 target_spacing=TARGET_SPACING, target_size=TARGET_SIZE, transform=None):
        self.images_dir = images_dir
        self.annotation_dir = annotation_dir
        self.modalities = modalities
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.transform = transform
        self.samples = self._collect_samples(subject_ids)

    def _collect_samples(self, subject_ids):
        samples = []
        for sid in subject_ids:
            patient_id, study_id = sid.split("_")
            patient_dir = os.path.join(self.images_dir, patient_id)
            mod_paths = {}
            for mod in self.modalities:
                p = os.path.join(patient_dir, f"{sid}_{mod}.mha")
                if os.path.exists(p):
                    mod_paths[mod] = p
            if len(mod_paths) == len(self.modalities):
                label_path = os.path.join(self.annotation_dir, f"{sid}.nii.gz")
                samples.append({"mods": mod_paths, "label": label_path, "id": sid})
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_and_preprocess(self, path, is_label=False):
        """Read -> resample -> center crop/pad -> normalize (images only)."""
        image = sitk.ReadImage(path)
        image = resample_volume(image, self.target_spacing, is_label=is_label)
        vol = sitk.GetArrayFromImage(image).astype(np.float32)  # (D, H, W)
        vol = center_crop_or_pad(vol, self.target_size)
        if not is_label:
            vol = zscore_normalize(vol)
        return vol

    def __getitem__(self, idx):
        sample = self.samples[idx]

        channels = []
        for mod in self.modalities:
            vol = self._load_and_preprocess(sample["mods"][mod], is_label=False)
            channels.append(vol)
        image = np.stack(channels, axis=0)  # (C, D, H, W)

        if os.path.exists(sample["label"]):
            label = self._load_and_preprocess(sample["label"], is_label=True)
            label = (label > 0).astype(np.float32)
        else:
            label = np.zeros(self.target_size, dtype=np.float32)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label[np.newaxis]  # (C,D,H,W), (1,D,H,W)


class PICAIPreprocessedDataset(Dataset):
    """Load preprocessed NIfTI files, skipping online resampling."""

    def __init__(self, preprocessed_dir, subject_ids, transform=None):
        self.images_dir = os.path.join(preprocessed_dir, "images")
        self.labels_dir = os.path.join(preprocessed_dir, "labels")
        self.transform = transform
        self.samples = [sid for sid in subject_ids
                        if os.path.exists(os.path.join(self.images_dir, f"{sid}.nii.gz"))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid = self.samples[idx]
        image = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.images_dir, f"{sid}.nii.gz"))
        ).astype(np.float32)  # (C, D, H, W)

        label = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.labels_dir, f"{sid}.nii.gz"))
        ).astype(np.float32)  # (D, H, W)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label[np.newaxis]  # (C,D,H,W), (1,D,H,W)
