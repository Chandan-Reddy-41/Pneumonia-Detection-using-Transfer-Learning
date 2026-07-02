"""
Phase 1: Data Loading & EDA

Expects the dataset unzipped at data/chest_xray/ with the structure:
    data/chest_xray/train/NORMAL, data/chest_xray/train/PNEUMONIA
    data/chest_xray/val/NORMAL,   data/chest_xray/val/PNEUMONIA
    data/chest_xray/test/NORMAL,  data/chest_xray/test/PNEUMONIA
"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from src import config


def count_images(directory):
    """Count images per class in a given split directory."""
    counts = {}
    for cls in config.CLASS_NAMES:
        cls_dir = os.path.join(directory, cls)
        counts[cls] = len(os.listdir(cls_dir)) if os.path.isdir(cls_dir) else 0
    return counts


def print_dataset_summary():
    """Print class counts for train/val/test splits."""
    print(f"{'SPLIT':<6} | {'NORMAL':>7} | {'PNEUMONIA':>9} | {'TOTAL':>6}")
    print("-" * 40)
    for split, path in [("train", config.TRAIN_DIR), ("val", config.VAL_DIR), ("test", config.TEST_DIR)]:
        counts = count_images(path)
        total = sum(counts.values())
        print(f"{split.upper():<6} | {counts.get('NORMAL', 0):>7} | {counts.get('PNEUMONIA', 0):>9} | {total:>6}")

    train_counts = count_images(config.TRAIN_DIR)
    if train_counts.get("NORMAL", 0) and train_counts.get("PNEUMONIA", 0):
        ratio = train_counts["PNEUMONIA"] / train_counts["NORMAL"]
        print(f"\nTrain set class imbalance ratio (PNEUMONIA:NORMAL) = {ratio:.2f}:1")
        if ratio > 1.5:
            print("-> Noticeable imbalance. Class weights will be applied during training (see preprocessing.py).")


def plot_class_distribution(save_path=None):
    """Bar chart of class counts across train/val/test."""
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
    """Show a grid of sample X-rays, one row per class."""
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


def load_datasets():
    """Load train/val/test as tf.data.Dataset objects using directory structure."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary",
        seed=config.SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary",
        seed=config.SEED,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    print(f"Class indices (0/1 mapping): {config.CLASS_NAMES}")
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    print_dataset_summary()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plot_class_distribution(save_path=os.path.join(config.RESULTS_DIR, "class_distribution.png"))
    plot_sample_images(config.TRAIN_DIR, save_path=os.path.join(config.RESULTS_DIR, "sample_xrays.png"))
