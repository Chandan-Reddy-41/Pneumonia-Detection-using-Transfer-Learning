"""Training loop shared by both the baseline CNN and the transfer-learning model."""
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from src import config


def get_callbacks(checkpoint_path, monitor="val_loss", mode="min", patience=8):
    """Return a list of sensible Keras callbacks for training.

    v1 monitored val_recall with patience=5, which -- combined with a
    16-image validation set -- gave EarlyStopping a flat, uninformative
    signal (val_recall stuck at exactly 0.0) and stopped the baseline CNN
    after 6 epochs before it had learned anything.

    Now that prepare_data_split.py gives a real, properly-sized validation
    set, val_recall would be usable again -- but val_loss is kept as the
    default here because it's a smoother, less noisy signal in the early
    epochs, particularly for the baseline CNN training from random
    initialization. Pass monitor="val_recall", mode="max" explicitly for
    models you expect to already be learning well (e.g. transfer learning).
    """
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


def train_model(
    model,
    train_ds,
    val_ds,
    class_weights,
    epochs=25,
    checkpoint_name="model.keras",
    monitor="val_loss",
    mode="min",
    patience=8,
):
    """Train `model` using the provided datasets and callbacks.

    Args:
        model: A compiled `tf.keras.Model` instance.
        train_ds: tf.data.Dataset for training.
        val_ds: tf.data.Dataset for validation.
        class_weights: Dict mapping class indices to weights.
        epochs: Number of epochs to train.
        checkpoint_name: Filename to save the best model to under `models/`.
        monitor: Metric to monitor for early stopping and checkpointing.
        mode: 'min' or 'max' depending on the monitored metric.
        patience: EarlyStopping patience.

    Returns:
        A `History` object from `model.fit()`.
    """
    # Ensure the models directory exists before writing checkpoints
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.MODELS_DIR, checkpoint_name)
    callbacks = get_callbacks(checkpoint_path, monitor=monitor, mode=mode, patience=patience)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    return history


def plot_history(history, model_name="model", save_path=None):
    """Plot training and validation metrics stored in a Keras History.

    Args:
        history: History object returned by `model.fit()`.
        model_name: String used in plot titles.
        save_path: Optional path to save the resulting figure (PNG).
    """
    # Standard metrics tracked during training; adapt if more metrics are added
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
