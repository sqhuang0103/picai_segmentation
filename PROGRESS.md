# 完成步骤记录

## Step 1: 项目初始化
- **日期**: 2026-07-27
- **状态**: 已完成
- **内容**: 创建项目目录结构，规划各模块职责

## Step 2: 下载数据
- **日期**: 2026-07-27
- **状态**: 进行中
- **内容**: `data/download_data.sh` — 从 Zenodo 下载 PI-CAI 公开影像（5 folds），从 GitHub 克隆 `picai_labels`

## Step 3: 配置文件
- **日期**: 2026-07-27
- **状态**: 已完成
- **内容**: `configs/config.py` — 对齐 PI-CAI baseline 设置：spacing 3.0×0.5×0.5, patch 20×256×256, batch=8, epoch=100, lr=1e-3, focal loss gamma=1.0, Adam+AMSGrad

## Step 4: 实现 UNet 网络
- **日期**: 2026-07-27
- **状态**: 已完成
- **内容**: `models/unet.py` — 3D UNet，InstanceNorm+LeakyReLU，各向异性 stride (2,2,2)→(1,2,2)→(1,2,2)→(1,2,2)→(2,2,2)，通道 32→64→128→256→512→1024

## Step 5: 数据预处理
- **日期**: 2026-07-28
- **状态**: 已完成
- **内容**: `data/dataset.py` — 重采样到统一 spacing → center crop/pad → 0.5/99.5 百分位 clip + z-score 归一化，标注用最近邻插值

## Step 6: 数据增强
- **日期**: 2026-07-28
- **状态**: 已完成
- **内容**: `data/augmentations.py` — nnUNet 风格增强：弹性形变(α=0~900, σ=9~13, p=0.2)、旋转(±30°, p=0.2)、缩放(0.7~1.4, p=0.2)、翻转、高斯噪声(p=0.1)、高斯模糊(p=0.2)、亮度(p=0.15)、gamma(0.7~1.5, p=0.3)

## Step 7: 损失函数
- **日期**: 2026-07-28
- **状态**: 已完成
- **内容**: `utils/losses.py` — Focal Loss (gamma=1.0, alpha=inverse class balance) + Dice Loss + 组合 FocalDiceLoss

## Step 8: 评估指标
- **日期**: 2026-07-27
- **状态**: 已完成
- **内容**: `utils/metrics.py` — AUC、Dice、Sen@90Spe、Spe@90Sen

## Step 9: 训练流程
- **日期**: 2026-07-28
- **状态**: 已完成
- **内容**: `train.py` — Focal Loss + Adam(amsgrad=True) + 每10 epoch 验证，保存最优模型

## Step 10: 评估脚本
- **日期**: 2026-07-28
- **状态**: 已完成
- **内容**: `evaluate.py` — 加载 best_model.pth，计算并打印全部 4 个指标

## Step 11: 离线预处理
- **日期**: 2026-07-28
- **状态**: 已完成
- **内容**: `data/preprocess.py` — 原始 .mha → 重采样+裁剪+归一化 → .nii.gz，多进程并行处理全部 1500 例；`preprocess.sh` — SGE 提交脚本（8 core, 4h）；`PICAIPreprocessedDataset` — 直接读取预处理后的 NIfTI

## 后续 TODO
- [ ] 提交预处理任务 `qsub preprocess.sh`
- [ ] 预处理完成后提交训练 `qsub train.sh`
- [ ] 整理结果
