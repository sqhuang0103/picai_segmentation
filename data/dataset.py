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
    """Resample a SimpleITK image to target spacing.
    Returns: resampled SimpleITK image with size (W', H', D') in sitk order."""
    original_spacing = np.array(image.GetSpacing())       # (W, H, D) in sitk
    original_size = np.array(image.GetSize())              # (W, H, D) in sitk
    target_spacing_sitk = np.array(target_spacing)[::-1]  # (D,H,W) -> (W,H,D)

    new_size = np.round(original_size * original_spacing / target_spacing_sitk).astype(int).tolist()  # (W', H', D')

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
    """Center crop or zero-pad a 3D numpy array to target_shape.
    Args: vol (D, H, W), target_shape (D', H', W')
    Returns: (D', H', W')"""
    result = np.zeros(target_shape, dtype=vol.dtype)  # (D', H', W')
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
    return result  # (D', H', W')


def zscore_normalize(vol, low_pct=CLIP_LOW_PERCENTILE, high_pct=CLIP_HIGH_PERCENTILE):
    """Percentile clipping (0.5/99.5) + instance-wise z-score normalization.
    Args: vol (D, H, W)
    Returns: (D, H, W) normalized"""
    low = np.percentile(vol, low_pct)
    high = np.percentile(vol, high_pct)
    vol = np.clip(vol, low, high)  # (D, H, W)
    mean = vol.mean()
    std = vol.std() + 1e-8
    return (vol - mean) / std  # (D, H, W)


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
        vol = center_crop_or_pad(vol, self.target_size)  # (D, H, W)
        if not is_label:
            vol = zscore_normalize(vol)  # (D, H, W)
        return vol  # (D, H, W)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        channels = []
        for mod in self.modalities:
            vol = self._load_and_preprocess(sample["mods"][mod], is_label=False)  # (D, H, W)
            channels.append(vol)
        image = np.stack(channels, axis=0)  # (C, D, H, W)

        if os.path.exists(sample["label"]):
            label = self._load_and_preprocess(sample["label"], is_label=True)  # (D, H, W)
            label = (label > 0).astype(np.float32)  # (D, H, W)
        else:
            label = np.zeros(self.target_size, dtype=np.float32)  # (D, H, W)

        if self.transform:
            image, label = self.transform(image, label)  # (C, D, H, W), (D, H, W)

        return image, label[np.newaxis]  # (C, D, H, W), (1, D, H, W)


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

        return image, label[np.newaxis]  # (C, D, H, W), (1, D, H, W)


class PICAIMultiViewDataset(Dataset):
    """
    Multi-view dataset: returns K random spatial crops per volume for training,
    or a single center crop for deterministic inference.
    """

    def __init__(self, preprocessed_dir, subject_ids, crop_size=(16, 192, 192),
                 num_views=4, is_train=True, transform=None):
        self.images_dir = os.path.join(preprocessed_dir, "images")
        self.labels_dir = os.path.join(preprocessed_dir, "labels")
        self.crop_size = crop_size
        self.num_views = num_views
        self.is_train = is_train
        self.transform = transform
        self.samples = [sid for sid in subject_ids
                        if os.path.exists(os.path.join(self.images_dir, f"{sid}.nii.gz"))]

    def __len__(self):
        return len(self.samples)

    def _random_crop(self, image, label):
        """Draw a random spatial crop from the volume.
        Args: image (C, D, H, W), label (D, H, W)
        Returns: (C, cd, ch, cw), (cd, ch, cw)"""
        _, D, H, W = image.shape
        cd, ch, cw = self.crop_size
        d0 = np.random.randint(0, max(D - cd, 0) + 1)
        h0 = np.random.randint(0, max(H - ch, 0) + 1)
        w0 = np.random.randint(0, max(W - cw, 0) + 1)
        return (image[:, d0:d0+cd, h0:h0+ch, w0:w0+cw],  # (C, cd, ch, cw)
                label[d0:d0+cd, h0:h0+ch, w0:w0+cw])      # (cd, ch, cw)

    def _center_crop(self, image, label):
        """Center crop for deterministic inference.
        Args: image (C, D, H, W), label (D, H, W)
        Returns: (C, cd, ch, cw), (cd, ch, cw)"""
        _, D, H, W = image.shape
        cd, ch, cw = self.crop_size
        d0 = max((D - cd) // 2, 0)
        h0 = max((H - ch) // 2, 0)
        w0 = max((W - cw) // 2, 0)
        return (image[:, d0:d0+cd, h0:h0+ch, w0:w0+cw],  # (C, cd, ch, cw)
                label[d0:d0+cd, h0:h0+ch, w0:w0+cw])      # (cd, ch, cw)

    def __getitem__(self, idx):
        sid = self.samples[idx]
        image = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.images_dir, f"{sid}.nii.gz"))
        ).astype(np.float32)  # (C, D, H, W)
        label = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.labels_dir, f"{sid}.nii.gz"))
        ).astype(np.float32)  # (D, H, W)

        if self.is_train:
            views_img, views_lbl = [], []
            for _ in range(self.num_views):
                ci, cl = self._random_crop(image, label)  # (C, cd, ch, cw), (cd, ch, cw)
                if self.transform:
                    ci, cl = self.transform(ci, cl)  # (C, cd, ch, cw), (cd, ch, cw)
                views_img.append(ci)
                views_lbl.append(cl)
            return np.stack(views_img), np.stack(views_lbl)  # (K, C, cd, ch, cw), (K, cd, ch, cw)
        else:
            ci, cl = self._center_crop(image, label)  # (C, cd, ch, cw), (cd, ch, cw)
            return ci, cl[np.newaxis]  # (C, cd, ch, cw), (1, cd, ch, cw)
