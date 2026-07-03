"""Central configuration: paths, image settings, and class names.

This module centralizes filesystem locations and model input shapes so
that other modules can import a single source of truth. Keeping these
values here makes it easier to change dataset layout or model sizes
without scattering literals across the codebase.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Baseline CNN trains from scratch, so a smaller input keeps it fast on CPU.
BASELINE_IMG_SIZE = (150, 150)
BASELINE_INPUT_SHAPE = (150, 150, 3)

# MobileNetV2 was pretrained at 224x224 -- matching that resolution matters
# for transfer-learning quality (feeding it 150x150 was a mismatch in v1).
TRANSFER_IMG_SIZE = (224, 224)
TRANSFER_INPUT_SHAPE = (224, 224, 3)

BATCH_SIZE = 32
SEED = 42

# validation_split used when carving val out of the merged train+val pool
VAL_SPLIT = 0.15

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chest_xray")
TRAIN_DIR = os.path.join(DATA_DIR, "train")  # after prepare_data_split.py, this is train+val merged
VAL_DIR = os.path.join(DATA_DIR, "val")       # kept for reference / rollback only, unused after the split fix
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Index 0 = NORMAL, Index 1 = PNEUMONIA (alphabetical order, matches
# tf.keras.utils.image_dataset_from_directory's default label assignment)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
