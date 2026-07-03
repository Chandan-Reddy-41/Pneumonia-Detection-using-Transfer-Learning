"""
Fixes the validation-set problem: the original Kaggle chest_xray dataset's
val/ folder has only 16 images (8 NORMAL, 8 PNEUMONIA). That's far too few
to give meaningful signal to EarlyStopping/ModelCheckpoint -- val accuracy
can only take values like 0/16, 1/16, ..., which is why training curves in
v1 showed flat, jumpy lines and the baseline CNN's early stopping triggered
on noise instead of real signal.

This script merges val/ into train/ so a proper validation_split can be
carved out later (via image_dataset_from_directory's validation_split
argument, done in data_loader.py). test/ is left untouched.

Run once, before main.py:
    python prepare_data_split.py
"""
import os
import shutil
import logging

from src import config

logger = logging.getLogger(__name__)


def merge_val_into_train():
    """Move images from the tiny `val/` split into `train/`.

    The original dataset's `val/` contains only 16 images; this merges
    those images into `train/` so a proper stratified validation split
    can be carved out later via `image_dataset_from_directory(..., validation_split=...)`.
    """

    if not os.path.isdir(config.VAL_DIR):
        logger.info("No val/ folder found at %s -- nothing to merge, skipping.", config.VAL_DIR)
        return

    moved = 0
    for cls in config.CLASS_NAMES:
        src_dir = os.path.join(config.VAL_DIR, cls)
        dst_dir = os.path.join(config.TRAIN_DIR, cls)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            if os.path.isfile(src_path) and not os.path.exists(dst_path):
                shutil.move(src_path, dst_path)
                moved += 1

    logger.info("Moved %d images from val/ into train/.", moved)
    held_out = f"{config.VAL_SPLIT:.0%}"
    logger.info(
        "A proper stratified validation split will now be carved out of train/ at load time (%s held out).",
        held_out,
    )

    # Leave an empty val/ folder in place (harmless) rather than deleting it,
    # in case anything else references the path.


if __name__ == "__main__":
    merge_val_into_train()
