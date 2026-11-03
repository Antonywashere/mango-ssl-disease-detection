# Self-Supervised Learning for Early Mango Leaf Stress & Disease Detection

A single Jupyter notebook that pretrains an image encoder on **unlabeled mango leaf images** using self-supervised learning (SimCLR or BYOL), then uses it for 8-class disease classification and an embedding-based early-stress score. Mango only.

## What's in the notebook

| Stage | What it does |
|---|---|
| Data | Downloads the Kaggle mango leaf dataset via `kagglehub`, with a synthetic fallback if the download fails |
| Leak-safe split | MD5-hashes every image so exact duplicates are never split across train/val/test |
| SSL pretraining | **SimCLR** (NT-Xent) or **BYOL** (EMA target network), selectable with `SSL_METHOD` |
| Backbone | ResNet-18 or ResNet-50, selectable with `BACKBONE` |
| Collapse check | Reports embedding diversity so a collapsed encoder is caught early |
| Downstream | Linear probe, full fine-tune, and **XGBoost** on frozen embeddings |
| Diagnostics | Train vs. validation loss plots, confusion matrices, classification reports |
| Visualization | t-SNE of the learned embeddings |
| Early stress scoring | Distance from the "healthy" centroid in embedding space |

Every training stage **checkpoints after each epoch and resumes automatically**. Checkpoints are named with `RUN_TAG = {SSL_METHOD}_{BACKBONE}`, so SimCLR and BYOL runs never overwrite each other.

## Dataset

Default: [`binhpt310/mango-combined-dataset`](https://www.kaggle.com/datasets/binhpt310/mango-combined-dataset) (8 classes: Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Healthy, Powdery Mildew, Sooty Mould).

This dataset is built by merging several sources and augmenting heavily, so many images are near-duplicates of each other. The notebook only removes **exact** duplicates; near-duplicates (for example, a flipped copy of the same photo) can still land in different splits and make accuracy look better than real-world performance. Treat the numbers accordingly, and test on photos from a different source if you can.

The original [`aryashah2k/mango-leaf-disease-dataset`](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset) (about 4,000 images) also works: change `KAGGLE_DATASET_SLUG` in Section 2.

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib pillow tqdm kagglehub xgboost jupyter
```

Install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) if you have an NVIDIA GPU. Training on CPU is impractically slow at full epoch counts.

### Kaggle credentials

1. Kaggle → Account → **Create New API Token** to download `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`), or set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.
3. Never commit this file (it is already in `.gitignore`).

## Running

```bash
jupyter notebook mango_ssl_stress_disease_detection.ipynb
```

Run the cells top to bottom. Key settings to adjust:

| Setting | Default | Notes |
|---|---|---|
| `SSL_METHOD` | `"simclr"` | `"simclr"` or `"byol"` |
| `BACKBONE` | `"resnet18"` | `"resnet18"` or `"resnet50"` |
| `EPOCHS_SSL` | `150` | Use 10–20 first as a smoke test |
| `SSL_BATCH_SIZE` | `128` | Lower to 64 or 32 on GPU out-of-memory errors |
| `DOWNSTREAM_BATCH_SIZE` | `64` | Classifier train/val/test batch size |
| `LINEAR_PROBE_EPOCHS` / `FINETUNE_EPOCHS` | `30` / `20` | Downstream training epochs |
| `NUM_WORKERS` | `0` | Raise carefully; worker processes can hang in Jupyter on Windows |

### Stopping and resuming

Re-run any training cell and it resumes from the last saved epoch. To start a stage fresh, delete its file in `checkpoints/`. If you change the model architecture, delete the old checkpoints too, since their saved weights will no longer match.

### Comparing SimCLR and BYOL

Run the whole notebook once per method. After switching `SSL_METHOD`, re-run from the pretraining cell downward so the downstream stages use the new encoder. Compare **linear probe** accuracy rather than fine-tuned accuracy, since it more directly measures representation quality.

## Results

Fill these in from your own runs.

| SSL method | Backbone | Linear probe acc. | Fine-tune acc. | XGBoost acc. |
|---|---|---|---|---|
| SimCLR | ResNet-18 | 97.5% | 96.9% | 98.3% |
| BYOL | ResNet-18 | 97.8% | 96.6% | 97.6% |

## Repository layout

```
.
├── mango_ssl_stress_disease_detection.ipynb
└── README.md
```


## Limitations

- The "early stress" score is a heuristic (distance from the healthy centroid with a fixed threshold), and the dataset has no early-stage labels, so it is not validated against real early-stage disease.
- Accuracy on a studio-style, augmented dataset may not transfer to field photos with different lighting and backgrounds.
- No leaf segmentation or lighting normalization is applied beyond standard resizing and ImageNet normalization.

