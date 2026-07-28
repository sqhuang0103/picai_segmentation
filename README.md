# PI-CAI Challenge: Prostate Cancer Segmentation

## Project Structure

```
picai_segmentation/
├── README.md                # Project overview (this file)
├── PROGRESS.md              # Step-by-step progress log
├── configs/
│   ├── config.py            # Hyperparameters and path configuration
│   └── splits.json          # Official 5-fold cross-validation splits (patient-level)
├── data/
│   ├── download_data.sh     # Data download script (Zenodo + picai_labels)
│   ├── preprocess.py        # Offline preprocessing: .mha → resample + crop + normalize → .nii.gz
│   ├── dataset.py           # Dataset classes (online and preprocessed)
│   └── augmentations.py     # nnUNet-style data augmentation
├── models/
│   └── unet.py              # Custom 3D UNet (shared by baseline and multi-view)
├── utils/
│   ├── losses.py            # Focal Loss + Dice Loss
│   └── metrics.py           # Evaluation metrics
├── train.py                 # Training script (baseline UNet)
├── train_multiview.py       # Training script (multi-view UNet)
├── evaluate.py              # Evaluation script (baseline UNet)
├── evaluate_multiview.py    # Evaluation script (multi-view UNet)
└── outputs/                 # Model checkpoints and logs
```

## Task

- **Objective**: Clinically significant prostate cancer (csPCa) lesion segmentation
- **Challenge**: [PI-CAI (Prostate Imaging: Cancer AI)](https://pi-cai.grand-challenge.org/)
- **Network**: Custom 3D UNet (self-implemented)
- **Input**: Multi-parametric MRI — T2W + ADC + HBV (3 channels)

## Preprocessing (following PI-CAI baseline)

- Resample to 3.0 × 0.5 × 0.5 mm voxel spacing
- Center crop / pad to 20 × 256 × 256 voxels
- Intensity clipping at 0.5 / 99.5 percentiles + instance-wise z-score normalization
- Labels resampled with nearest-neighbor interpolation

## Network Architecture

### Baseline: 3D UNet
- Encoder-decoder with skip connections
- InstanceNorm3d + LeakyReLU
- Feature channels: 32 → 64 → 128 → 256 → 512 → 1024 (bottleneck)
- Anisotropic pooling strides: (2,2,2) → (1,2,2) → (1,2,2) → (1,2,2) → (2,2,2)

### Experiment: Multi-View Training Strategy
- During training, K random spatial crops (views) are drawn from each volume per forward pass
- Each view has its own spatially corresponding label; per-view focal losses are averaged
- A patient-level consistency regularisation penalises variance of mean predicted probability across views:
  `L_total = (1/K) · Σ L_focal(logits_k, label_k) + λ · Var(mean_prob_per_view)`
- λ is linearly warmed up from 0 over a configurable number of epochs
- At inference, the center crop is used for deterministic predictions
- Same UNet backbone; the multi-view logic is purely a training strategy

## Training Configuration (following PI-CAI baseline)

| Parameter | Value |
|-----------|-------|
| Loss | Focal Loss (gamma=1.0, alpha=inverse class balance) |
| Optimizer | Adam + AMSGrad |
| Learning Rate | 1e-3 |
| Batch Size | 8 |
| Epochs | 100 |
| Validation | Every 10 epochs |
| Augmentation | nnUNet-style (elastic deformation, rotation, scaling, flipping, Gaussian noise/blur, brightness, gamma) |

## Evaluation Metrics

- **AUC** (Area Under the ROC Curve) — patient-level
- **Dice Coefficient** — voxel-level
- **Sensitivity @ 90% Specificity** — patient-level
- **Specificity @ 90% Sensitivity** — patient-level

## Usage

```bash
# 1. Download data from Zenodo and clone annotations
bash data/download_data.sh

# 2. Offline preprocessing (resample + crop + normalize → NIfTI)
qsub preprocess.sh          # or: python data/preprocess.py

# 3a. Train baseline UNet
qsub train.sh               # or: python train.py

# 3b. Train multi-view UNet
qsub train_multiview.sh     # or: python train_multiview.py

# 4. Evaluate
python evaluate.py           # baseline
python evaluate_multiview.py # multi-view
```

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- SimpleITK
- scikit-learn
- SciPy
