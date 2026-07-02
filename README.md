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
```

## Usage

```bash
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
| Transfer Learning (MobileNetV2) | — | — | — | — | — |

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
