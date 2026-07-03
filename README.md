# Pneumonia Detection from Chest X-Rays

A CNN-based image classifier that detects pneumonia from pediatric chest X-ray
images, comparing a custom-built CNN against a MobileNetV2 transfer-learning
model, with Grad-CAM explainability to visualize model decisions.

## Problem Statement

Pneumonia is diagnosed via visual inspection of chest X-rays by radiologists.
This project explores whether a CNN can flag likely pneumonia cases from
X-ray images, and — just as importantly — whether the model's decisions are
interpretable enough to trust.

Because a missed pneumonia case (false negative) is more costly than a false
alarm, **recall on the PNEUMONIA class** is treated as the primary metric
alongside accuracy, not accuracy alone.

## Dataset

Chest X-Ray dataset (pediatric patients), organized as:

```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

> The `data/` folder is git-ignored (dataset too large for version control).
> Download the dataset and unzip it into `data/chest_xray/` before running
> anything.

## Approach

| Phase | Description |
|---|---|
| 1 | Data loading & EDA — class distribution, sample image visualization |
| 2 | Preprocessing — normalization, augmentation, class-weight balancing |
| 3 | Baseline CNN — custom 4-block Conv2D architecture trained from scratch |
| 4 | Transfer learning — MobileNetV2 (ImageNet weights) + custom head |
| 5 | Evaluation — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, side-by-side comparison |
| 6 | Explainability — Grad-CAM heatmaps showing what the model attends to |

### Why MobileNetV2 for transfer learning?

Lightweight enough to train and fine-tune on a CPU while still leveraging
strong ImageNet-pretrained features — a better fit for local (non-GPU)
training than VGG16/ResNet.

### Why class weights instead of oversampling?

The training set is imbalanced (more PNEUMONIA than NORMAL images). Class
weights integrate directly into `model.fit()` without needing to duplicate
or synthesize images, and work cleanly with `tf.data` pipelines.

### Why no vertical flip in augmentation?

A vertically flipped chest X-ray is anatomically invalid — it would teach
the model orientations that don't occur in real data.

## Project Structure

```
pneumonia-detection/
├── data/chest_xray/       # dataset (not included — see Setup)
├── models/                # saved model checkpoints (.keras, git-ignored)
├── results/                # plots, confusion matrices, comparison table
├── src/
│   ├── config.py           # paths & constants
│   ├── data_loader.py       # Phase 1: loading & EDA
│   ├── preprocessing.py     # Phase 2: augmentation, class weights
│   ├── models.py            # Phase 3 & 4: architectures
│   ├── train.py              # training loop & callbacks
│   ├── evaluate.py           # Phase 5: metrics & comparison
│   └── gradcam.py            # Phase 6: Grad-CAM implementation
├── main.py                  # runs Phases 1-5 end to end
├── gradcam_demo.py           # runs Phase 6 (after main.py)
├── prepare_data_split.py     # run once before main.py -- fixes the tiny val set
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone this repo
git clone <your-repo-url>
cd pneumonia-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset and unzip into data/chest_xray/
#    (should produce data/chest_xray/train, /val, /test)

# 5. Fix the validation split (the original val/ folder has only 16 images --
#    too few for EarlyStopping to get a meaningful signal). This merges it
#    into train/ so a proper stratified split can be carved out at load time.
python prepare_data_split.py
```

## Usage

```bash
# One-time: fix the validation split (skip if you already ran it during setup)
python prepare_data_split.py

# Run the full pipeline: EDA -> preprocessing -> both models -> evaluation
python main.py

# After main.py completes, run Grad-CAM explainability
python gradcam_demo.py
```

Outputs land in `results/`: class distribution plot, sample X-ray grid,
training curves, confusion matrices, ROC comparison, model comparison CSV,
and Grad-CAM overlay.

Trained model checkpoints land in `models/` (best weights by validation
recall, via `ModelCheckpoint`).

## Results

*(Fill in after running `main.py` on your machine — copy values from
`results/model_comparison.csv`.)*

| Model | Accuracy | Precision (Pneumonia) | Recall (Pneumonia) | F1 (Pneumonia) | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN | — | — | — | — | — |
| Transfer Learning (MobileNetV2, fine-tuned) | — | — | — | — | — |

### v1 run (for reference — see Changelog below for what was fixed)

| Model | Accuracy | Precision (Pneumonia) | Recall (Pneumonia) | F1 (Pneumonia) | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN | 0.375 | 0.00 | 0.00 | 0.00 | 0.605 |
| Transfer Learning (MobileNetV2) | 0.756 | 0.73 | 0.96 | 0.83 | 0.889 |

## Changelog

**v3** — fixes based on analyzing the v2 run:
- **Baseline CNN learning rate raised from 1e-4 to 1e-3.** In v2 the
  baseline's train accuracy sat flat at ~50% for all 9 epochs -- the
  validation-split fix solved the *measurement* problem, but the model
  still wasn't learning. 1e-4 is an appropriate fine-tuning rate for the
  pretrained transfer model, but too conservative to move a randomly
  initialized network within a handful of epochs.
- **Threshold-tuned metrics now land in `model_comparison.csv` as their own
  row**, with their own confusion matrix (`cm_transfer_tuned.png`), instead
  of only being printed to console and easy to lose. v2's transfer model
  hit AUC=0.943 but collapsed to predicting PNEUMONIA for almost every
  image at the default 0.5 threshold (206/234 NORMAL images misclassified)
  -- the precision-recall curve showed a much better operating point was
  available, this just makes it visible in the saved results.

**v2** — fixes based on analyzing the v1 run:
- **Fixed the validation set.** The original Kaggle `val/` folder has only
  16 images, which gave `EarlyStopping`/`ModelCheckpoint` a flat,
  near-meaningless signal (val_recall stuck at exactly 0.0). This is what
  caused the v1 baseline CNN to collapse to predicting NORMAL for every
  image (37.5% accuracy = exactly the NORMAL class proportion). `val/` is
  now merged into `train/` (`prepare_data_split.py`) and a proper
  stratified split is carved out via `validation_split` at load time.
- **Baseline CNN now monitors `val_loss` instead of `val_recall`** for
  early stopping (patience 8, up from 5) — a smoother signal for a model
  training from random initialization.
- **Transfer learning now uses 224×224 input** (`config.TRANSFER_IMG_SIZE`),
  matching the resolution MobileNetV2 was pretrained at. v1 used 150×150,
  a resolution mismatch that likely cost feature quality.
- **Added a fine-tuning pass** (`models.unfreeze_for_finetuning`) — after
  initial frozen-backbone training, the top layers of MobileNetV2 unfreeze
  for a second pass at a much lower learning rate (1e-5).
- **Added threshold tuning** (`evaluate.find_best_threshold`,
  `plot_precision_recall_curve`) — the default 0.5 decision threshold is
  rarely optimal; this sweeps thresholds to maximize F1 while keeping
  recall on PNEUMONIA at or above 90%.

## Explainability

Grad-CAM heatmaps (see `results/gradcam_sample.png`) overlay the regions of
the X-ray the model weighted most heavily in its prediction — used to sanity
check that the model is attending to lung fields rather than incidental
artifacts (text markers, equipment edges, etc.).

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, pandas, matplotlib

## Author

Moola Chandan Reddy
[GitHub](https://github.com/Chandan-Reddy-41) · chandureddymoola@gmail.com

## License

MIT — see [LICENSE](LICENSE)
