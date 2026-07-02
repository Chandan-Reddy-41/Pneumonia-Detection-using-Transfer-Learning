"""
Training loop shared by both the baseline CNN and the transfer-learning model.
"""
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from src import config


def get_callbacks(checkpoint_path, monitor="val_recall", mode="max", patience=5):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode=mode, patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor=monitor, mode=mode, save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
    ]


def train_model(model, train_ds, val_ds, class_weights, epochs=25, checkpoint_name="model.keras"):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.MODELS_DIR, checkpoint_name)
    callbacks = get_callbacks(checkpoint_path)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    return history


def plot_history(history, model_name="model", save_path=None):
    metrics = ["loss", "accuracy", "recall", "auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    for ax, m in zip(axes, metrics):
        ax.plot(history.history[m], label="train")
        ax.plot(history.history[f"val_{m}"], label="val")
        ax.set_title(f"{model_name}: {m}")
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
