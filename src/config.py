"""
Central configuration: paths, image settings, class names.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_SIZE = (150, 150)
INPUT_SHAPE = (150, 150, 3)
BATCH_SIZE = 32
SEED = 42

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chest_xray")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Index 0 = NORMAL, Index 1 = PNEUMONIA (alphabetical order, matches
# tf.keras.utils.image_dataset_from_directory's default label assignment)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
