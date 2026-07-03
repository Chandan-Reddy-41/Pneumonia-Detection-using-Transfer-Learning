"""
Pneumonia Detection from Chest X-Rays — End-to-End Pipeline (v2)
===================================================================

Changes from v1 (see README "Changelog" section for the full story):
- Fixed the 16-image validation set (was giving EarlyStopping/ModelCheckpoint
  meaningless, flat signal -- caused the baseline CNN to collapse to
  predicting one class for every image).
- Transfer learning now uses 224x224 input (matches MobileNetV2's
  pretraining resolution; v1 used a mismatched 150x150).
- Added a second fine-tuning pass for the transfer model (unfreezes the
  top of the MobileNetV2 backbone at a low learning rate).
- Added threshold tuning (Phase 5b) since the default 0.5 cutoff is rarely
  the best operating point for a recall-sensitive screening task.

Before running:
1. Download and unzip the dataset so it sits at:
     data/chest_xray/train/{NORMAL,PNEUMONIA}
     data/chest_xray/val/{NORMAL,PNEUMONIA}
     data/chest_xray/test/{NORMAL,PNEUMONIA}
2. pip install -r requirements.txt
3. python prepare_data_split.py   <-- run this once, before main.py

Usage:
    python main.py

For Phase 6 (Grad-CAM), run gradcam_demo.py after this completes.
"""
import os
import logging

from src import config, data_loader, preprocessing, models, train, evaluate


def main():
    """Run the full training and evaluation pipeline.

    This function orchestrates the end-to-end phases described in the
    module docstring: data loading, preprocessing, baseline training,
    transfer learning (frozen and fine-tuned), evaluation, and threshold
    tuning. It creates results and models directories if missing and
    writes plots and CSV comparison outputs to `results/`.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("%s", "=" * 60)
    logger.info("PHASE 1: Data Loading & EDA")
    logger.info("%s", "=" * 60)
    data_loader.print_dataset_summary()

    logger.info("\n%s", "=" * 60)
    logger.info("PHASE 2: Preprocessing")
    logger.info("%s", "=" * 60)
    class_weights = preprocessing.get_class_weights()
    logger.info("Class weights (to counter imbalance): %s", class_weights)

    # ---- Baseline CNN: 150x150, rescaled to [0,1] ----
    train_raw_base, val_raw_base, test_raw_base = data_loader.load_datasets(config.BASELINE_IMG_SIZE)
    train_ds_base = preprocessing.prepare_dataset(train_raw_base, training=True, rescale=True)
    val_ds_base = preprocessing.prepare_dataset(val_raw_base, training=False, rescale=True)
    test_ds_base = preprocessing.prepare_dataset(test_raw_base, training=False, rescale=True)

    logger.info("\n%s", "=" * 60)
    logger.info("PHASE 3: Baseline CNN (from scratch)")
    logger.info("%s", "=" * 60)
    # lr=1e-3 (Adam's standard default), not 1e-4: a from-scratch randomly
    # initialized network needs a higher learning rate to start moving.
    # 1e-4 is a fine-tuning rate, appropriate for the pretrained transfer
    # model but too conservative here -- v2's baseline sat flat at ~50%
    # train accuracy for 9 epochs because of this.
    baseline = models.compile_model(models.build_baseline_cnn(), lr=1e-3)
    baseline.summary()
    baseline_history = train.train_model(
        baseline, train_ds_base, val_ds_base, class_weights,
        epochs=25, checkpoint_name="baseline_cnn.keras",
        monitor="val_loss", mode="min", patience=8,  # smoother signal for a from-scratch model
    )
    train.plot_history(
        baseline_history, "Baseline CNN",
        save_path=os.path.join(config.RESULTS_DIR, "history_baseline.png"),
    )

    # ---- Transfer learning: 224x224, MobileNetV2's own preprocessing ----
    train_raw_tl, val_raw_tl, test_raw_tl = data_loader.load_datasets(config.TRANSFER_IMG_SIZE)
    train_ds_tl = preprocessing.prepare_dataset(train_raw_tl, training=True, rescale=False)
    val_ds_tl = preprocessing.prepare_dataset(val_raw_tl, training=False, rescale=False)
    test_ds_tl = preprocessing.prepare_dataset(test_raw_tl, training=False, rescale=False)

    logger.info("\n%s", "=" * 60)
    logger.info("PHASE 4a: Transfer Learning -- frozen backbone")
    logger.info("%s", "=" * 60)
    transfer_model, base_model = models.build_transfer_model()
    transfer_model = models.compile_model(transfer_model, lr=1e-4)
    transfer_model.summary()
    frozen_history = train.train_model(
        transfer_model, train_ds_tl, val_ds_tl, class_weights,
        epochs=15, checkpoint_name="transfer_mobilenetv2_frozen.keras",
        monitor="val_recall", mode="max", patience=6,  # already learning fast, recall signal is meaningful here
    )
    train.plot_history(
        frozen_history, "Transfer Learning (frozen)",
        save_path=os.path.join(config.RESULTS_DIR, "history_transfer_frozen.png"),
    )

    logger.info("\n%s", "=" * 60)
    logger.info("PHASE 4b: Transfer Learning -- fine-tuning top layers")
    logger.info("%s", "=" * 60)
    models.unfreeze_for_finetuning(base_model, fine_tune_at=100)
    transfer_model = models.compile_model(transfer_model, lr=1e-5)  # much lower LR for fine-tuning
    finetune_history = train.train_model(
        transfer_model, train_ds_tl, val_ds_tl, class_weights,
        epochs=10, checkpoint_name="transfer_mobilenetv2.keras",
        monitor="val_recall", mode="max", patience=5,
    )
    train.plot_history(
        finetune_history, "Transfer Learning (fine-tuned)",
        save_path=os.path.join(config.RESULTS_DIR, "history_transfer_finetuned.png"),
    )

    logger.info("\n%s", "=" * 60)
    logger.info("PHASE 5: Evaluation & Comparison")
    logger.info("%s", "=" * 60)
    baseline_results = evaluate.evaluate_model(
        baseline,
        test_ds_base,
        "Baseline CNN",
    )
    transfer_results = evaluate.evaluate_model(
        transfer_model,
        test_ds_tl,
        "Transfer Learning (MobileNetV2, fine-tuned)",
    )

    evaluate.plot_confusion_matrix(
        baseline_results["confusion_matrix"],
        "Baseline CNN",
        save_path=os.path.join(
            config.RESULTS_DIR,
            "cm_baseline.png",
        ),
    )
    evaluate.plot_confusion_matrix(
        transfer_results["confusion_matrix"], "Transfer Learning",
        save_path=os.path.join(config.RESULTS_DIR, "cm_transfer.png"),
    )
    evaluate.plot_roc_curves(
        [baseline_results, transfer_results],
        save_path=os.path.join(
            config.RESULTS_DIR, "roc_comparison.png",
        ),
    )

    logger.info("\n%s", "=" * 60)
    logger.info("PHASE 5b: Threshold Tuning (Transfer Learning model)")
    logger.info("%s", "=" * 60)
    evaluate.plot_precision_recall_curve(
        transfer_results,
        save_path=os.path.join(config.RESULTS_DIR, "pr_curve_transfer.png"),
    )
    best = evaluate.find_best_threshold(transfer_results, min_recall=0.90)
    logger.info("Suggested threshold (best F1 with recall >= 0.90): %s", best)

    # Recompute full metrics at the tuned threshold -- not just the printed
    # console line -- so this becomes its own row in the comparison table
    # and gets its own confusion matrix, instead of getting lost in logs.
    tuned_results = evaluate.metrics_at_threshold(transfer_results, best["threshold"])
    evaluate.plot_confusion_matrix(
        tuned_results["confusion_matrix"], tuned_results["model_name"],
        save_path=os.path.join(config.RESULTS_DIR, "cm_transfer_tuned.png"),
    )

    # Final comparison table now includes all three: baseline, transfer at
    # default 0.5, and transfer at the tuned threshold -- so the improvement
    # from threshold tuning is visible directly in the CSV, not just console output.
    comparison = evaluate.comparison_table([baseline_results, transfer_results, tuned_results])
    logger.info("%s", comparison)
    comparison.to_csv(os.path.join(config.RESULTS_DIR, "model_comparison.csv"), index=False)

    logger.info("\nDone. Trained models saved in models/, plots and comparison table in results./")
    logger.info("Next: run `python gradcam_demo.py` for Phase 6 explainability visualizations.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
