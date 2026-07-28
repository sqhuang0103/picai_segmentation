# PI-CAI Challenge: Prostate Cancer Segmentation

## 项目结构

```
picai_segmentation/
├── README.md                # 项目结构与说明（本文件）
├── PROGRESS.md              # 完成步骤记录
├── configs/
│   ├── config.py            # 超参数与路径配置
│   └── splits.json          # 官方 5-fold 划分（patient-level）
├── data/
│   ├── download_data.sh     # 数据下载脚本（Zenodo + picai_labels）
│   ├── preprocess.py        # 离线预处理：.mha → 重采样+裁剪+归一化 → .nii.gz
│   ├── dataset.py           # 数据加载（支持在线/离线预处理）
│   └── augmentations.py     # nnUNet 风格数据增强
├── models/
│   └── unet.py              # 自定义 3D UNet
├── utils/
│   ├── losses.py            # Focal Loss + Dice Loss
│   └── metrics.py           # 评估指标
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
└── outputs/                 # 模型权重与日志
```

## 任务

- **目标**: 前列腺癌临床显著性病灶分割（PI-CAI Challenge）
- **网络**: 自实现 3D UNet
- **输入**: T2W + ADC + HBV 三模态 MRI

## 预处理（对齐 PI-CAI baseline）

- 重采样至 3.0 × 0.5 × 0.5 mm spacing
- 中心裁剪/填充至 20 × 256 × 256 体素
- 0.5/99.5 百分位 clip + instance-wise z-score 归一化

## 训练设置（对齐 PI-CAI baseline）

| 参数 | 值 |
|------|-----|
| Loss | Focal Loss (gamma=1.0, alpha=inverse class balance) |
| Optimizer | Adam + AMSGrad |
| Learning Rate | 1e-3 |
| Batch Size | 8 |
| Epochs | 100 |
| Normalization | InstanceNorm3d + LeakyReLU |
| Augmentation | nnUNet 风格（弹性形变、旋转、缩放、翻转、噪声、模糊、gamma） |

## 评估指标

- AUC — patient-level
- Dice Coefficient — voxel-level
- Sensitivity @ 90% Specificity — patient-level
- Specificity @ 90% Sensitivity — patient-level

## 使用方法

```bash
bash data/download_data.sh        # 1. 下载数据
qsub preprocess.sh                # 2. 离线预处理 → .nii.gz
qsub train.sh                     # 3. 训练
python evaluate.py                # 4. 评估
```

## 运行环境

- Python 3.9+
- PyTorch, NumPy, SimpleITK, scikit-learn, SciPy
