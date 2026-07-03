"""Phase 3 and 4: Model definitions.

Contains baseline CNN and MobileNetV2 transfer model builders used by the
training and evaluation scripts.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from src import config


def build_baseline_cnn(input_shape=config.BASELINE_INPUT_SHAPE):
    """Build a compact CNN for training from scratch.

    Architecture: 4 convolutional blocks with batch normalization and
    max-pooling, followed by global average pooling and a small dense head.

    Args:
        input_shape: Tuple describing input image shape, e.g. (150, 150, 3).

    Returns:
        An uncompiled `tf.keras.Model`.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="baseline_cnn",
    )
    return model


def build_transfer_model(input_shape=config.TRANSFER_INPUT_SHAPE):
    """Build a MobileNetV2-based model with a custom classification head.

    The MobileNetV2 backbone is loaded with ImageNet weights and frozen by default.
    The returned tuple is `(model, base_model)` where `base_model` refers to the
    MobileNetV2 instance; this allows selective unfreezing for a second
    fine-tuning pass.

    Args:
        input_shape: Model input shape, typically `config.TRANSFER_INPUT_SHAPE`.

    Returns:
        (model, base_model)
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="transfer_mobilenetv2")
    return model, base_model


def unfreeze_for_finetuning(base_model, fine_tune_at=100):
    """Unfreeze a suffix of the backbone for fine-tuning.

    Args:
        base_model: The pretrained backbone model (MobileNetV2 instance).
        fine_tune_at: Integer layer index from which layers will be set to
            trainable; layers before this index remain frozen.

    Returns:
        The modified `base_model` with updated `.trainable` flags.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return base_model


def compile_model(model, lr=1e-4):
    """Compile model with Adam optimizer and diagnostic-focused metrics.

    Args:
        model: Uncompiled Keras model.
        lr: Learning rate for the Adam optimizer.

    Returns:
        The compiled model (same object passed in).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
