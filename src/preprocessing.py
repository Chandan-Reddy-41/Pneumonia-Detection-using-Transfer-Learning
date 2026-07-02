"""
Phase 2: Preprocessing & Augmentation

- Normalizes pixel values
- Applies augmentation to the training set only
  (no vertical flip -- would produce an anatomically invalid X-ray)
- Computes class weights to counter the NORMAL/PNEUMONIA imbalance
- Builds efficient tf.data pipelines (cache + prefetch)
"""
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.data_loader import count_images

AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(0.1),
    ],
    name="data_augmentation",
)


def prepare_dataset(ds, training=False, rescale=True):
    """
    Apply normalization, (optionally) augmentation, and performance tuning
    to a raw tf.data.Dataset from image_dataset_from_directory.

    Set rescale=False when the model itself has a preprocessing layer
    baked in (e.g. tf.keras.applications.mobilenet_v2.preprocess_input),
    to avoid double-normalizing.
    """
    if rescale:
        ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(1000, seed=config.SEED)

    return ds.cache().prefetch(buffer_size=AUTOTUNE)


def get_class_weights():
    """
    Compute balanced class weights from the training set folder counts.
    Returns a dict like {0: weight_for_NORMAL, 1: weight_for_PNEUMONIA}.
    """
    counts = count_images(config.TRAIN_DIR)
    labels = np.concatenate(
        [
            np.zeros(counts["NORMAL"]),   # class 0 = NORMAL
            np.ones(counts["PNEUMONIA"]),  # class 1 = PNEUMONIA
        ]
    )
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes.astype(int), weights))
