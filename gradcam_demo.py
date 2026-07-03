"""
Phase 6: Grad-CAM Explainability Demo

Run this after main.py has trained and saved a model.
Prints model.summary() first so you can confirm the LAST_CONV_LAYER name
matches your model (MobileNetV2's final conv layer is "Conv_1"; if you
change architectures, update this).
"""
import os
import numpy as np
import tensorflow as tf
import logging

from src import config
from src.gradcam import visualize_gradcam

MODEL_PATH = os.path.join(config.MODELS_DIR, "transfer_mobilenetv2.keras")
LAST_CONV_LAYER = "Conv_1"  # MobileNetV2's last conv layer name


def load_sample_image(img_path):
    """Load a single image and return the model-ready array and original image.

    Args:
        img_path: Path to the image file to load.

    Returns:
        Tuple `(img_array, orig_img)` where `img_array` is a 4D numpy
        array shaped (1, H, W, C) suitable for model.predict(), and
        `orig_img` is the original image as an array for visualization.
    """
    img = tf.keras.utils.load_img(img_path, target_size=config.TRANSFER_IMG_SIZE)
    orig_img = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(orig_img, axis=0)
    return img_array, orig_img


def main():
    """Load a trained model and generate a Grad-CAM visualization.

    Expects a trained MobileNetV2-based model saved at `models/transfer_mobilenetv2.keras`.
    The function selects a sample image from the test set and writes a composite
    visualization (orig, heatmap, overlay) to `results/`.
    """

    # Ensure a trained model exists at the expected path
    logger = logging.getLogger(__name__)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python main.py` first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    sample_dir = os.path.join(config.TEST_DIR, "PNEUMONIA")
    sample_file = sorted(os.listdir(sample_dir))[0]
    img_path = os.path.join(sample_dir, sample_file)
    logger.info("\nGenerating Grad-CAM for: %s", img_path)

    img_array, orig_img = load_sample_image(img_path)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    visualize_gradcam(
        model, img_array, orig_img, LAST_CONV_LAYER,
        save_path=os.path.join(config.RESULTS_DIR, "gradcam_sample.png"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
