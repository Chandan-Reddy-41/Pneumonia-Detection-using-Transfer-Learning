"""Phase 5: Evaluation & Model Comparison.

Recall on the PNEUMONIA class is treated as the headline metric alongside
accuracy: in a diagnostic context a false negative (missed pneumonia) is
more costly than a false positive.
"""
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

from src import config

logger = logging.getLogger(__name__)


def evaluate_model(model, test_ds, model_name="model"):
    """Evaluate a trained model on a test dataset and return common metrics.

    Args:
        model: A trained `tf.keras.Model`.
        test_ds: A `tf.data.Dataset` yielding (image, label) pairs.
        model_name: Friendly name for printing and returned dict.

    Returns:
        A dict containing accuracy, precision, recall, f1, auc, confusion matrix,
        and raw `y_true`/`y_pred_prob` arrays for downstream plotting.
    """
    # Extract ground-truth labels and predicted probabilities
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel().astype(int)
    y_pred_prob = model.predict(test_ds).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )
    auc = roc_auc_score(y_true, y_pred_prob)
    cm = confusion_matrix(y_true, y_pred)

    logger.info("--- %s ---", model_name)
    logger.info("%s", classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0))
    logger.info("ROC-AUC: %.4f\n", auc)

    return {
        "model_name": model_name,
        "accuracy": report["accuracy"],
        "precision": report["PNEUMONIA"]["precision"],
        "recall": report["PNEUMONIA"]["recall"],
        "f1": report["PNEUMONIA"]["f1-score"],
        "auc": auc,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred_prob": y_pred_prob,
    }


def plot_confusion_matrix(cm, model_name, save_path=None):
    """Plot or save a confusion matrix for a model's predictions.

    Args:
        cm: Confusion matrix array (2x2 for binary classification).
        model_name: Title label for the plot.
        save_path: Optional path to save the figure as PNG.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(results_list, save_path=None):
    """Plot ROC curves for a list of result dicts produced by `evaluate_model`.

    Args:
        results_list: Iterable of result dicts containing `y_true`, `y_pred_prob`, and `auc`.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=(6, 6))
    for res in results_list:
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_pred_prob"])
        plt.plot(fpr, tpr, label=f"{res['model_name']} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_precision_recall_curve(res, save_path=None):
    """Plot precision-recall curve for a single model's prediction distribution.

    Args:
        res: Result dict from `evaluate_model` containing `y_true`, `y_pred_prob`, and current precision/recall.
        save_path: Optional path to save the figure.

    Returns:
        Tuple `(precisions, recalls, thresholds)` as returned by `sklearn.metrics.precision_recall_curve`.
    """
    precisions, recalls, thresholds = precision_recall_curve(res["y_true"], res["y_pred_prob"])
    plt.figure(figsize=(6, 6))
    plt.plot(recalls, precisions, label=res["model_name"])
    plt.axhline(
        res["precision"], color="gray", linestyle=":", alpha=0.5,
        label="current (0.5 threshold) precision",
    )
    plt.axvline(res["recall"], color="gray", linestyle="--", alpha=0.5, label="current (0.5 threshold) recall")
    plt.xlabel("Recall (Pneumonia)")
    plt.ylabel("Precision (Pneumonia)")
    plt.title(f"Precision-Recall Trade-off: {res['model_name']}")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return precisions, recalls, thresholds


def find_best_threshold(res, min_recall=0.90):
    """Suggest a classification threshold that maximizes F1 while keeping recall.

    Suggests a classification threshold that maximizes F1 while keeping
    recall on PNEUMONIA at or above `min_recall`. The default 0.5 threshold
    is rarely optimal -- for a screening tool you usually want to hold
    recall high and use the threshold sweep to buy back some precision
    without sacrificing missed-case rate.

    Returns a dict with the suggested threshold and the metrics at that
    threshold, so you can decide whether it's a better operating point
    than the default 0.5.
    """
    precisions, recalls, thresholds = precision_recall_curve(res["y_true"], res["y_pred_prob"])
    # precision_recall_curve returns one more point than thresholds; align them
    precisions, recalls = precisions[:-1], recalls[:-1]

    valid = recalls >= min_recall
    if not valid.any():
        logger.info("No threshold achieves recall >= %s; returning default 0.5.", min_recall)
        return {"threshold": 0.5, "precision": res["precision"], "recall": res["recall"]}

    f1s = np.where(valid, 2 * precisions * recalls / (precisions + recalls + 1e-8), -1)
    best_idx = np.argmax(f1s)

    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1s[best_idx]),
    }


def metrics_at_threshold(res, threshold):
    """Recompute metrics and confusion matrix at a custom decision threshold.

    Recomputes accuracy/precision/recall/F1/confusion matrix at a custom
    decision threshold (evaluate_model always uses the default 0.5).
    AUC is threshold-independent so it's carried over unchanged.

    Use this to turn a suggestion from find_best_threshold() into a full
    result dict -- e.g. to add it as its own row in comparison_table() and
    plot its own confusion matrix, instead of the tuned numbers only
    existing as a printed console line.
    """
    y_true = res["y_true"]
    y_pred_prob = res["y_pred_prob"]
    y_pred = (y_pred_prob >= threshold).astype(int)

    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "model_name": f"{res['model_name']} @ threshold={threshold:.2f}",
        "accuracy": report["accuracy"],
        "precision": report["PNEUMONIA"]["precision"],
        "recall": report["PNEUMONIA"]["recall"],
        "f1": report["PNEUMONIA"]["f1-score"],
        "auc": res["auc"],  # unchanged -- AUC doesn't depend on threshold
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred_prob": y_pred_prob,
    }


def comparison_table(results_list):
    """Build a pandas DataFrame summarizing results for comparison.

    Args:
        results_list: Iterable of result dicts returned by `evaluate_model`.

    Returns:
        `pandas.DataFrame` with summary metrics per model.
    """
    rows = []
    for res in results_list:
        rows.append(
            {
                "Model": res["model_name"],
                "Accuracy": round(res["accuracy"], 4),
                "Precision (Pneumonia)": round(res["precision"], 4),
                "Recall (Pneumonia)": round(res["recall"], 4),
                "F1 (Pneumonia)": round(res["f1"], 4),
                "ROC-AUC": round(res["auc"], 4),
            }
        )
    return pd.DataFrame(rows)
