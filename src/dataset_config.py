# Dataset-specific configuration for Cityscapes pipeline
# To adapt for a new dataset, edit only this file.

# === Paths ===
CITYSCAPES_ROOT = "/home/senai/Documentos/edge_segmentation/datasets/cityscapes"
IMG_DIR = f"{CITYSCAPES_ROOT}/leftImg8bit_trainvaltest/leftImg8bit"
MASK_DIR = f"{CITYSCAPES_ROOT}/gtFine_trainvaltest/gtFine"

# Suffixes for images and masks
IMAGE_SUFFIX = "_leftImg8bit.png"
MASK_SUFFIX = "_gtFine_labelIds.png"

# Output folders (relative to project root)
OUTPUTS_DIR = "outputs"
PLOTS_DIR = f"{OUTPUTS_DIR}/plots"
OVERLAYS_DIR = f"{OUTPUTS_DIR}/overlays"
STATISTICS_DIR = f"{OUTPUTS_DIR}/statistics"

# Prepared dataset structure
PREPARED_ROOT = "dataset_prepared/cityscapes"
PREPARED_SPLITS = ["train", "val", "test"]
PREPARED_IMAGES = "images"
PREPARED_MASKS = "masks"
PREPARED_METADATA = "metadata"

# Labels module (for dynamic import)
LABELS_MODULE = "helpers.cityscapes_labels"

# Overlay generation limit (None for all)
OVERLAY_MAX_IMAGES = 50
