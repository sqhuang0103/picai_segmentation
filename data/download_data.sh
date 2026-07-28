#!/bin/bash
# PI-CAI public data download script
# Images: Zenodo record 6624726
# Labels: github.com/DIAGNijmegen/picai_labels

set -e

# ============ Path configuration ============
DATA_ROOT="/home/rmapshu/Scratch/picai_data"
IMAGES_DIR="${DATA_ROOT}/images"
LABELS_DIR="${DATA_ROOT}/picai_labels"
ZIP_DIR="${DATA_ROOT}/zips"

mkdir -p "$IMAGES_DIR" "$ZIP_DIR"

# ============ Download images (5 folds) ============
echo "=== Downloading PI-CAI images ==="
ZENODO_BASE="https://zenodo.org/api/records/6624726/files"

for fold in 0 1 2 3 4; do
    FILENAME="picai_public_images_fold${fold}.zip"
    if [ -f "${ZIP_DIR}/${FILENAME}" ]; then
        echo "[skip] ${FILENAME} already exists"
    else
        echo "[download] ${FILENAME} ..."
        curl -C - -L "${ZENODO_BASE}/${FILENAME}/content" --output "${ZIP_DIR}/${FILENAME}"
    fi
done

# ============ Extract ============
echo "=== Extracting images ==="
for fold in 0 1 2 3 4; do
    FILENAME="picai_public_images_fold${fold}.zip"
    echo "[extract] ${FILENAME} ..."
    unzip -n "${ZIP_DIR}/${FILENAME}" -d "$IMAGES_DIR"
done

# ============ Download labels ============
echo "=== Downloading labels ==="
if [ -d "$LABELS_DIR" ]; then
    echo "[skip] picai_labels already exists"
else
    git clone https://github.com/DIAGNijmegen/picai_labels "$LABELS_DIR"
fi

# ============ Create symlinks to project directory ============
PROJ_DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
ln -sfn "$IMAGES_DIR" "${PROJ_DATA_DIR}/images"
ln -sfn "$LABELS_DIR" "${PROJ_DATA_DIR}/picai_labels"

echo ""
echo "=== Done ==="
echo "Images: ${IMAGES_DIR}"
echo "Labels: ${LABELS_DIR}"
echo "Symlinks created in: ${PROJ_DATA_DIR}/"
