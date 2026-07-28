import os
import json

# ============ 数据路径 ============
DATA_ROOT = os.environ.get("PICAI_DATA_ROOT", "/home/rmapshu/Scratch/picai_data")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
LABELS_DIR = os.path.join(DATA_ROOT, "picai_labels")
ANNOTATION_DIR = os.path.join(LABELS_DIR, "csPCa_lesion_delineations", "human_expert", "resampled")

PREPROCESSED_DIR = os.path.join(DATA_ROOT, "preprocessed")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 官方 5-fold 划分 ============
SPLITS_PATH = os.path.join(os.path.dirname(__file__), "splits.json")
with open(SPLITS_PATH) as f:
    SPLITS = json.load(f)
FOLD = 0  # 当前使用的 fold

# ============ 数据设置 ============
MODALITIES = ["t2w", "adc", "hbv"]
IN_CHANNELS = len(MODALITIES)
NUM_CLASSES = 2
TARGET_SPACING = (3.0, 0.5, 0.5)
IMAGE_SIZE = (20, 256, 256)

# ============ 网络结构 (PI-CAI baseline UNet) ============
BASE_FEATURES = 32
MODEL_FEATURES = [32, 64, 128, 256, 512, 1024]

# ============ 训练超参数 (PI-CAI baseline) ============
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
VALIDATE_N_EPOCHS = 10

# Focal Loss
FOCAL_LOSS_GAMMA = 1.0
POSITIVE_RATIO = 220.0 / 1500.0
