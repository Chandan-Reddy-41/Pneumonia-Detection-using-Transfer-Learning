"""Phase 1: Data Loading & EDA.

Expects the dataset unzipped at data/chest_xray/ with the structure:
    data/chest_xray/train/NORMAL, data/chest_xray/train/PNEUMONIA
    data/chest_xray/val/NORMAL,   data/chest_xray/val/PNEUMONIA
    data/chest_xray/test/NORMAL,  data/chest_xray/test/PNEUMONIA
"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

from src import config

logger = logging.getLogger(__name__)


def count_images(directory):
    """Count images per class in a given split directory.

    Args:
        directory: Path to a split directory that contains class subfolders
            matching `config.CLASS_NAMES` (e.g., "NORMAL", "PNEUMONIA").

    Returns:
        Dict mapping class name to integer count of files present.
    """
    counts = {}
    for cls in config.CLASS_NAMES:
        cls_dir = os.path.join(directory, cls)
        counts[cls] = len(os.listdir(cls_dir)) if os.path.isdir(cls_dir) else 0
    return counts


def print_dataset_summary():
    """Print class counts and imbalance ratio for train/val/test splits.

    This utility prints a small table to the console and computes the
    PNEUMONIA:NORMAL ratio on the training set to hint whether class
    weighting should be applied.
    """
    logger.info("%s", f"{'SPLIT':<6} | {'NORMAL':>7} | {'PNEUMONIA':>9} | {'TOTAL':>6}")
    logger.info("%s", "-" * 40)
    for split, path in [("train", config.TRAIN_DIR), ("val", config.VAL_DIR), ("test", config.TEST_DIR)]:
        counts = count_images(path)
        total = sum(counts.values())
        logger.info(
            "%s",
            (
                f"{split.upper():<6} | {counts.get('NORMAL', 0):>7} | "
                f"{counts.get('PNEUMONIA', 0):>9} | {total:>6}"
            ),
        )

    train_counts = count_images(config.TRAIN_DIR)
    if train_counts.get("NORMAL", 0) and train_counts.get("PNEUMONIA", 0):
        ratio = train_counts["PNEUMONIA"] / train_counts["NORMAL"]
        logger.info("\nTrain set class imbalance ratio (PNEUMONIA:NORMAL) = %.2f:1", ratio)
        if ratio > 1.5:
            logger.info(
                "-> Noticeable imbalance. Class weights will be applied during training "
                "(see preprocessing.py)."
            )


def plot_class_distribution(save_path=None):
    """Create and optionally save a bar chart showing class counts.

    Args:
        save_path: Optional file path to save the figure as PNG.
    """
    splits = ["train", "val", "test"]
    dirs = [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, split, d in zip(axes, splits, dirs):
        counts = count_images(d)
        ax.bar(counts.keys(), counts.values(), color=["#4C9AFF", "#FF6B6B"])
        ax.set_title(f"{split.capitalize()} Set")
        ax.set_ylabel("Number of Images")
        for i, v in enumerate(counts.values()):
            ax.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_images(directory, n_per_class=4, save_path=None):
    """Display sample images from `directory` for quick visual inspection.

    Args:
        directory: Path to a split directory with class subfolders.
        n_per_class: Number of images to show per class.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6))
    for row, cls in enumerate(config.CLASS_NAMES):
        cls_dir = os.path.join(directory, cls)
        files = sorted(os.listdir(cls_dir))[:n_per_class]
        for col, fname in enumerate(files):
            img_path = os.path.join(cls_dir, fname)
            img = tf.keras.utils.load_img(img_path, color_mode="grayscale")
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(cls, fontsize=10)
            axes[row, col].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def load_datasets(img_size):
    """Load train/val/test as tf.data.Dataset objects at the given image size.

    The training and validation sets are carved from the merged
    `train/` directory using a `validation_split`. Make sure
    `prepare_data_split.py` has been run to merge the tiny original
    `val/` folder into `train/` before calling this function.

    Args:
        img_size: Tuple (H, W) target size for images.

    Returns:
        Tuple `(train_ds, val_ds, test_ds)` of `tf.data.Dataset` objects.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=img_size,
        batch_size=config.BATCH_SIZE,
        label_mode="binary",
        validation_split=config.VAL_SPLIT,
        subset="training",
        seed=config.SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=img_size,
        batch_size=config.BATCH_SIZE,
        label_mode="binary",
        validation_split=config.VAL_SPLIT,
        subset="validation",
        seed=config.SEED,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        image_size=img_size,
        batch_size=config.BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    logger.info("Class indices (0/1 mapping): %s", config.CLASS_NAMES)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    print_dataset_summary()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plot_class_distribution(save_path=os.path.join(config.RESULTS_DIR, "class_distribution.png"))
    plot_sample_images(config.TRAIN_DIR, save_path=os.path.join(config.RESULTS_DIR, "sample_xrays.png"))
    logger.info(
        "\nNote: run `python prepare_data_split.py` first if you haven't "
        "-- it merges the tiny original val/ folder into train/ so a proper "
        "stratified validation split can be carved out at load time."
    )
