"""
Phase 5: Evaluation & Model Comparison

Recall on the PNEUMONIA class is treated as the headline metric alongside
accuracy: in a diagnostic context a false negative (missed pneumonia) is
more costly than a false positive.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from src import config


def evaluate_model(model, test_ds, model_name="model"):
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel().astype(int)
    y_pred_prob = model.predict(test_ds).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )
    auc = roc_auc_score(y_true, y_pred_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"--- {model_name} ---")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0))
    print(f"ROC-AUC: {auc:.4f}\n")

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(results_list, save_path=None):
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


def comparison_table(results_list):
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
