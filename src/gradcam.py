"""Phase 6: Grad-CAM Explainability.

Generates a heatmap showing which regions of the X-ray most influenced
the model's prediction. Useful for sanity-checking that the model is
actually attending to lung fields rather than spurious artifacts
(text markers, edges, equipment).
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Create a Grad-CAM heatmap for a single image and model.

    Args:
        img_array: 4D numpy array or tensor shaped (1, H, W, C) containing the input image.
        model: A `tf.keras.Model` instance for which to compute Grad-CAM.
        last_conv_layer_name: Name of the model's last convolutional layer.
        pred_index: Optional index of the prediction class to compute gradients for.
            For binary models a value of 0 is appropriate since output shape is (1,).

    Returns:
        A 2D numpy array containing the normalized heatmap in [0, 1].
    """
    # Build a model that maps the input image to the activations
    # of the last conv layer as well as the model's output.
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0
        class_channel = predictions[:, pred_index]

    # Compute gradients of the top predicted class w.r.t. the conv layer outputs
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weight the convolution outputs by the pooled gradients and collapse to heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(orig_img, heatmap, alpha=0.4):
    """Overlay a colorized heatmap on top of the original image.

    Args:
        orig_img: HxWxC numpy array (uint8 or float) of the original image.
        heatmap: 2D numpy array of the same spatial ratio as the model's conv output.
        alpha: Float opacity for the heatmap overlay.

    Returns:
        A PIL Image with the heatmap superimposed on the original.
    """
    # Convert normalized heatmap to 0-255 and map through a colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((orig_img.shape[1], orig_img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Blend the colorized heatmap with the original image
    superimposed = jet_heatmap * alpha + orig_img
    return tf.keras.utils.array_to_img(superimposed)


def visualize_gradcam(model, img_array, orig_img, last_conv_layer_name, save_path=None):
    """Compute and display (or save) a three-panel Grad-CAM figure.

    The figure shows the original image, the raw Grad-CAM heatmap, and
    the heatmap overlaid on the original image.

    Args:
        model: Trained Keras model.
        img_array: 4D array suitable for model input.
        orig_img: Original image array for visualization.
        last_conv_layer_name: Name of last conv layer inside `model`.
        save_path: Optional path to save the generated figure.
    """
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    superimposed = overlay_heatmap(orig_img, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_img.astype("uint8"))
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
