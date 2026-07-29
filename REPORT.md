# Clinically Significant Prostate Cancer Segmentation on Multi-Parametric MRI

> 3D UNet baseline with multi-view consistency training on the PI-CAI challenge dataset — architecture, failure analysis, and revised training strategy.

**Fold 0 · 1500 cases · July 2026**

---

## 1. Introduction

Clinically significant prostate cancer (csPCa) is diagnosed through multi-parametric MRI (mpMRI) combining T2-weighted, apparent diffusion coefficient (ADC), and high b-value diffusion-weighted (HBV) sequences. The PI-CAI challenge provides 1500 annotated cases with expert-delineated lesion masks, offering a standardised benchmark for automated csPCa segmentation.

This report presents a self-implemented 3D UNet trained under two strategies — a single-view baseline and a multi-view consistency approach — alongside a diagnosis of critical training failures discovered during evaluation and the corrective measures applied.

## 2. Data & Preprocessing

The PI-CAI public training set contains 1500 cases from multiple acquisition centres, split into 5 patient-level folds. All experiments use Fold 0 (1200 train / 300 validation). Each case provides three MRI sequences: T2W, ADC, and HBV.

### Preprocessing Pipeline

- Resample all sequences to spacing `3.0 × 0.5 × 0.5 mm`
- Centre crop or zero-pad to a fixed volume of `20 × 256 × 256` voxels
- Intensity clipping at 0.5 / 99.5 percentiles followed by instance-wise z-score normalisation
- Labels resampled with nearest-neighbour interpolation and binarised (csPCa vs. background)

### Class Distribution

> ⚠️ Only **47 / 300** validation cases (15.7%) contain annotated lesions. Within positive cases, lesion voxels typically occupy < 0.2% of the volume — a severe dual imbalance at both case and voxel level.

## 3. Network Architecture

A 3D UNet following the PI-CAI baseline architecture: encoder–decoder with skip connections, InstanceNorm3d + LeakyReLU (slope 0.01), and anisotropic pooling strides tuned to the non-cubic voxel spacing.

```
Input(3×20×256²) → Enc32(2,2,2) → Enc64(1,2,2) → Enc128(1,2,2) → Enc256(1,2,2) → Enc512(2,2,2)
    → Bottleneck(1024) → Dec(512→32) → Output(2×20×256²)
```

Encoder stages are annotated with output channels and pooling stride. Decoder mirrors the encoder with transposed convolutions and skip connections.

### Data Augmentation

nnUNet-style online augmentation:

- Elastic deformation (α ∈ [0, 900], σ ∈ [9, 13], p = 0.2)
- Rotation (±30°, p = 0.2)
- Scaling (0.7–1.4×, p = 0.2)
- Random flips along all axes
- Gaussian noise (p = 0.1), Gaussian blur (p = 0.2)
- Multiplicative brightness (p = 0.15), Gamma transform (0.7–1.5, p = 0.3)

## 4. Training Strategy

### Baseline

| Parameter | Value |
|---|---|
| Loss | Focal Loss (γ = 1.0, α = 1 − positive ratio) |
| Optimiser | Adam + AMSGrad |
| Learning rate | 1 × 10⁻³ (fixed) |
| Batch size | 8 |
| Epochs | 100 |
| Validation | Every 10 epochs |
| Sampling | Uniform random |

### Multi-View Consistency

Each training volume yields K = 4 random spatial crops of size 16 × 192 × 192. Each crop is paired with its own spatially corresponding label; per-view losses are averaged. A patient-level consistency term penalises variance of mean predicted probability across views:

```
L_total = (1/K) · Σ L_seg(logits_k, label_k) + λ · (1/K) · Σ (p̄_k − p̄)²
```

λ is linearly warmed up from 0 to 1.0 over the first 10 epochs. At inference, a single deterministic centre crop is used. Batch size is reduced to 2 (each sample contains 4 views).

## 5. Initial Results (v1)

The baseline completed 100 epochs; best checkpoint was saved at epoch 10. The initially reported metrics appeared promising:

| Metric | Value |
|---|---|
| AUC | 0.9231 |
| Dice (reported) | 0.8333 ⚠️ **MISLEADING** |
| Sensitivity @ 90% Specificity | 0.8609 |
| Specificity @ 90% Sensitivity | 0.8310 |

## 6. Failure Analysis

Per-case inspection revealed the model had **not learned to segment lesions at all**. Every one of the 47 lesion-positive cases scored a true Dice of 0.0000.

> ❌ **All 47 lesion-positive cases have voxel-level Dice = 0.** The model predicted empty masks for 46 of them. The one case with non-zero prediction (10032) had 979 predicted voxels, but they occupied a completely different spatial region from the 5620 ground-truth voxels — zero intersection.

### How Dice = 0.833 Was Produced

The `compute_dice` function uses a smoothing term (ε = 10⁻⁵):

```
dice = (2 · intersection + ε) / (pred.sum + target.sum + ε)
```

When both prediction and ground truth are empty — as in all 253 lesion-free cases — the formula returns (0 + ε) / (0 + 0 + ε) ≈ 1.0. The mean Dice across 300 cases is therefore dominated by 253 trivial "perfect" scores: 253 / 300 ≈ 0.843.

### Root Causes

| # | Issue | Effect | Severity |
|---|---|---|---|
| 1 | Focal Loss only — no Dice Loss | Voxel-level CE with `reduction="mean"` averages over all 1.3M voxels per volume; lesion voxels at ~0.15% are drowned. The model minimises loss by predicting all-background. | **CRITICAL** |
| 2 | No positive-case oversampling | With 84% negative cases, most mini-batches contain zero lesion voxels — no gradient signal for learning foreground patterns. | **CRITICAL** |
| 3 | Smoothed Dice metric hides failures | Empty-vs-empty Dice ≈ 1.0 inflates reported mean, masking that all lesion cases score exactly 0. | **HIGH** |
| 4 | Fixed learning rate (1 × 10⁻³, 100 epochs) | Severe overfitting: val Dice peaked at epoch 10, then degraded steadily (0.83 → 0.55 by epoch 100). | **MODERATE** |

### Per-Case Breakdown (Validation Set, Fold 0)

| Category | Count | Model Behaviour |
|---|---|---|
| True Negatives (no lesion, no prediction) | 250 | Correct — trivially |
| False Negatives (lesion present, no prediction) | 46 | Complete miss, Dice = 0 |
| False Negatives (lesion present, prediction elsewhere) | 1 | Predicted in wrong region, Dice = 0 |
| False Positives (no lesion, prediction present) | 3 | Spurious predictions |

## 7. Corrective Measures (v2)

Four targeted corrections address the identified root causes. The same fixes are applied to both the baseline and multi-view training scripts.

| | Before (v1) | After (v2) |
|---|---|---|
| Loss | Focal Loss only | Focal Loss + Dice Loss combined |
| Sampling | Uniform random | WeightedRandomSampler (~50:50 pos/neg) |
| Val metric | Smoothed Dice (all cases) | Strict Dice on lesion-positive cases only |
| LR schedule | Fixed 10⁻³ | Cosine annealing 10⁻³ → 10⁻⁶ |

### 7.1 Combined Focal + Dice Loss

Dice Loss directly optimises spatial overlap and is inherently robust to class imbalance — it normalises by the union of prediction and target, so the background majority cannot dominate. The combined loss preserves Focal Loss's voxel-level classification signal while adding the region-level optimisation objective:

```
L_seg = L_focal + L_dice
```

### 7.2 Positive-Case Oversampling

A `WeightedRandomSampler` assigns inverse-frequency weights so that each epoch draws roughly equal numbers of lesion-positive and lesion-negative cases. This ensures every mini-batch exposes the network to foreground voxels.

### 7.3 Lesion-Only Dice Metric

A strict `compute_dice_strict` function (no smoothing) returns NaN for empty-vs-empty cases, which are then excluded from the mean. The new `Dice_lesion` metric reports only on cases where a lesion exists or a prediction was made — reflecting actual segmentation quality.

### 7.4 Cosine Annealing Schedule

The learning rate follows a cosine decay from 1 × 10⁻³ to 1 × 10⁻⁶ over 100 epochs, reducing the overfitting observed in v1 where a fixed high learning rate caused validation performance to degrade after epoch 10.

## 8. Multi-View Strategy — Early Signal

Before the v2 corrections were applied, the multi-view model (running the v1 loss) showed marginally higher Dice at epoch 10: **0.8433** vs. the baseline's 0.8333. Under the original (flawed) metric, this difference is negligible — both models were producing near-empty predictions inflated by the same smoothing artefact.

The multi-view consistency approach remains theoretically sound: by exposing the network to multiple spatial views of the same patient and penalising inconsistency, it encourages robust, spatially-aware representations. The v2 corrections — particularly the Dice Loss and oversampling — should enable this strategy to demonstrate meaningful improvement over the baseline.

## 9. Experimental Plan

- Re-run baseline training with v2 corrections (`qsub train.sh`)
- Re-run multi-view training with v2 corrections (`qsub train_multiview.sh`)
- Evaluate both with `Dice_lesion` as the primary metric
- Compare baseline vs. multi-view on all four metrics (AUC, Dice_lesion, Sen@90Spe, Spe@90Sen)
- Produce per-case visualisations of best/worst predictions for qualitative analysis

## 10. Project Structure

```
picai_segmentation/
├── configs/
│   ├── config.py              # Hyperparameters, paths, splits
│   └── splits.json            # Official 5-fold patient-level splits
├── data/
│   ├── download_data.sh       # Download from Zenodo + picai_labels
│   ├── preprocess.py          # Offline resampling + normalisation
│   ├── dataset.py             # Dataset classes (standard / multi-view)
│   └── augmentations.py       # nnUNet-style augmentation
├── models/
│   └── unet.py                # 3D UNet (encoder–decoder + skip connections)
├── utils/
│   ├── losses.py              # FocalLoss / DiceLoss / FocalDiceLoss
│   └── metrics.py             # AUC, Dice, Dice_lesion, Sen@Spe, Spe@Sen
├── train.py                   # Baseline training (v2)
├── train.sh                   # SGE job script — baseline
├── train_multiview.py         # Multi-view training (v2)
├── train_multiview.sh         # SGE job script — multi-view
├── evaluate.py                # Baseline evaluation
├── evaluate_multiview.py      # Multi-view evaluation
├── preprocess.sh              # SGE job script — preprocessing
└── outputs/                   # Checkpoints + logs (generated at runtime)
```
