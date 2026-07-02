"""
Phase 3: Baseline CNN (from scratch)
Phase 4: Transfer Learning (MobileNetV2)
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from src import config


def build_baseline_cnn(input_shape=config.INPUT_SHAPE):
    """A compact custom CNN: 4 conv blocks + dense classification head."""
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


def build_transfer_model(input_shape=config.INPUT_SHAPE, fine_tune_at=None):
    """
    MobileNetV2 backbone (ImageNet weights) + custom classification head.
    Chosen for being lightweight enough to fine-tune on CPU.

    fine_tune_at: if set (e.g. 100), unfreezes layers from this index onward
                  for a second fine-tuning pass. Leave None to keep the
                  backbone fully frozen (feature-extraction only).

    Note: expects raw images in [0, 255] -- MobileNetV2's own preprocessing
    is applied inside the model, so use prepare_dataset(..., rescale=False)
    for this model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = fine_tune_at is not None
    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

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


def compile_model(model, lr=1e-4):
    """Binary crossentropy + Adam, tracking accuracy/precision/recall/AUC.

    Recall is tracked explicitly because in a diagnostic setting a missed
    pneumonia case (false negative) is costlier than a false alarm.
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
