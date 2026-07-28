#!/bin/bash
# PI-CAI 公开数据下载脚本
# 数据来源: Zenodo record 6624726
# 标注来源: github.com/DIAGNijmegen/picai_labels

set -e

# ============ 配置路径 ============
DATA_ROOT="/home/rmapshu/Scratch/picai_data"
IMAGES_DIR="${DATA_ROOT}/images"
LABELS_DIR="${DATA_ROOT}/picai_labels"
ZIP_DIR="${DATA_ROOT}/zips"

mkdir -p "$IMAGES_DIR" "$ZIP_DIR"

# ============ 下载影像数据（5 folds） ============
echo "=== 下载 PI-CAI 影像数据 ==="
ZENODO_BASE="https://zenodo.org/api/records/6624726/files"

for fold in 0 1 2 3 4; do
    FILENAME="picai_public_images_fold${fold}.zip"
    if [ -f "${ZIP_DIR}/${FILENAME}" ]; then
        echo "[跳过] ${FILENAME} 已存在"
    else
        echo "[下载] ${FILENAME} ..."
        curl -C - -L "${ZENODO_BASE}/${FILENAME}/content" --output "${ZIP_DIR}/${FILENAME}"
    fi
done

# ============ 解压 ============
echo "=== 解压影像数据 ==="
for fold in 0 1 2 3 4; do
    FILENAME="picai_public_images_fold${fold}.zip"
    echo "[解压] ${FILENAME} ..."
    unzip -n "${ZIP_DIR}/${FILENAME}" -d "$IMAGES_DIR"
done

# ============ 下载标注 ============
echo "=== 下载标注数据 ==="
if [ -d "$LABELS_DIR" ]; then
    echo "[跳过] picai_labels 已存在"
else
    git clone https://github.com/DIAGNijmegen/picai_labels "$LABELS_DIR"
fi

# ============ 创建软链接到项目目录 ============
PROJ_DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
ln -sfn "$IMAGES_DIR" "${PROJ_DATA_DIR}/images"
ln -sfn "$LABELS_DIR" "${PROJ_DATA_DIR}/picai_labels"

echo ""
echo "=== 完成 ==="
echo "影像路径: ${IMAGES_DIR}"
echo "标注路径: ${LABELS_DIR}"
echo "项目软链接已创建在: ${PROJ_DATA_DIR}/"
