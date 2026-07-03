"""Phase 2: Preprocessing & Augmentation.

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
    """Prepare a tf.data.Dataset by normalizing, augmenting and optimizing.

    Args:
        ds: Raw dataset from `image_dataset_from_directory` yielding (image, label).
        training: If True, apply augmentation and shuffling.
        rescale: If True, scale images to [0, 1]. Set False for models that
            include their own preprocessing layer (e.g., MobileNetV2).

    Returns:
        A tf.data.Dataset optimized with `cache()` and `prefetch()`.
    """
    if rescale:
        ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(1000, seed=config.SEED)

    return ds.cache().prefetch(buffer_size=AUTOTUNE)


def get_class_weights():
    """Compute balanced class weights from counts in `config.TRAIN_DIR`.

    Returns a dict mapping integer class index to weight suitable for
    passing as `class_weight` to `model.fit()`.
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
