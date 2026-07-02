"""
Pneumonia Detection from Chest X-Rays — End-to-End Pipeline
=============================================================

Runs Phases 1-5: data loading, preprocessing, baseline CNN training,
transfer learning training, and evaluation/comparison.

Before running:
1. Download and unzip the dataset so it sits at:
     data/chest_xray/train/{NORMAL,PNEUMONIA}
     data/chest_xray/val/{NORMAL,PNEUMONIA}
     data/chest_xray/test/{NORMAL,PNEUMONIA}
2. pip install -r requirements.txt

Usage:
    python main.py

For Phase 6 (Grad-CAM), run gradcam_demo.py after this completes.
"""
import os

from src import config, data_loader, preprocessing, models, train, evaluate


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Data Loading & EDA")
    print("=" * 60)
    data_loader.print_dataset_summary()
    train_ds_raw, val_ds_raw, test_ds_raw = data_loader.load_datasets()

    print("\n" + "=" * 60)
    print("PHASE 2: Preprocessing")
    print("=" * 60)
    class_weights = preprocessing.get_class_weights()
    print("Class weights (to counter imbalance):", class_weights)

    # Baseline CNN pipeline: rescaled to [0,1]
    train_ds_base = preprocessing.prepare_dataset(train_ds_raw, training=True, rescale=True)
    val_ds_base = preprocessing.prepare_dataset(val_ds_raw, training=False, rescale=True)
    test_ds_base = preprocessing.prepare_dataset(test_ds_raw, training=False, rescale=True)

    # Transfer learning pipeline: raw [0,255], MobileNetV2 preprocesses internally
    train_ds_tl = preprocessing.prepare_dataset(train_ds_raw, training=True, rescale=False)
    val_ds_tl = preprocessing.prepare_dataset(val_ds_raw, training=False, rescale=False)
    test_ds_tl = preprocessing.prepare_dataset(test_ds_raw, training=False, rescale=False)

    print("\n" + "=" * 60)
    print("PHASE 3: Baseline CNN (from scratch)")
    print("=" * 60)
    baseline = models.compile_model(models.build_baseline_cnn())
    baseline.summary()
    baseline_history = train.train_model(
        baseline, train_ds_base, val_ds_base, class_weights,
        epochs=25, checkpoint_name="baseline_cnn.keras",
    )
    train.plot_history(
        baseline_history, "Baseline CNN",
        save_path=os.path.join(config.RESULTS_DIR, "history_baseline.png"),
    )

    print("\n" + "=" * 60)
    print("PHASE 4: Transfer Learning (MobileNetV2)")
    print("=" * 60)
    transfer_model, _base = models.build_transfer_model()
    transfer_model = models.compile_model(transfer_model)
    transfer_model.summary()
    transfer_history = train.train_model(
        transfer_model, train_ds_tl, val_ds_tl, class_weights,
        epochs=15, checkpoint_name="transfer_mobilenetv2.keras",
    )
    train.plot_history(
        transfer_history, "Transfer Learning",
        save_path=os.path.join(config.RESULTS_DIR, "history_transfer.png"),
    )

    print("\n" + "=" * 60)
    print("PHASE 5: Evaluation & Comparison")
    print("=" * 60)
    baseline_results = evaluate.evaluate_model(baseline, test_ds_base, "Baseline CNN")
    transfer_results = evaluate.evaluate_model(transfer_model, test_ds_tl, "Transfer Learning (MobileNetV2)")

    comparison = evaluate.comparison_table([baseline_results, transfer_results])
    print(comparison)
    comparison.to_csv(os.path.join(config.RESULTS_DIR, "model_comparison.csv"), index=False)

    evaluate.plot_confusion_matrix(
        baseline_results["confusion_matrix"], "Baseline CNN",
        save_path=os.path.join(config.RESULTS_DIR, "cm_baseline.png"),
    )
    evaluate.plot_confusion_matrix(
        transfer_results["confusion_matrix"], "Transfer Learning",
        save_path=os.path.join(config.RESULTS_DIR, "cm_transfer.png"),
    )
    evaluate.plot_roc_curves(
        [baseline_results, transfer_results],
        save_path=os.path.join(config.RESULTS_DIR, "roc_comparison.png"),
    )

    print("\nDone. Trained models saved in models/, plots and comparison table in results/.")
    print("Next: run `python gradcam_demo.py` for Phase 6 explainability visualizations.")


if __name__ == "__main__":
    main()
