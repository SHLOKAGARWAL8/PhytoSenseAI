# AgroDetect — Phase 1 & Phase 2 execution guide (PyTorch)

This update keeps the existing training notebooks as historical/research records and establishes `train_models.py` as the canonical reproducible pipeline.

## Phase 1 — Dataset and preprocessing

### 1. Environment (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pillow numpy scikit-learn matplotlib pandas jupyter
```

Use the PyTorch installation selector if your CUDA version differs: https://pytorch.org/get-started/locally/.

### 2. Download and prepare PlantVillage

The raw folder must contain the 38 class folders directly, or a parent folder with a `train` subfolder containing those class folders.

```powershell
python preprocess_dataset.py --raw-dir "D:\datasets\PlantVillage" --output-dir dataset --seed 42 --clean-output
```

This writes `dataset/train`, `dataset/val`, `dataset/test`, `class_indices.json`, `config.py`, `dataset_manifest.jsonl`, `preprocessing_metrics.json`, and `preprocessing_report.md`.

**Leakage control:** source files are SHA-256 hashed before splitting. Exact duplicates are assigned to one split only. Cross-class duplicate groups are listed for label review.

### 3. Verify the split

```powershell
python audit_dataset.py --dataset-dir dataset --output dataset_audit.json
```

Proceed only when the audit prints `"status": "PASS"`, the class mapping matches ImageFolder, all images are 224×224, and both `cross_split_duplicate_groups` and `cross_class_duplicate_groups` are zero. Resolve any cross-class label conflicts and rerun preprocessing/audit before training.

## Phase 2 — Four-model training and comparison

### 1. Run a quick pipeline smoke test

Use a small subset only for checking code execution, not for reporting research results. The full 38-class dataset is required for final results.

### 2. Train all four models

```powershell
python train_models.py --model all --data-dir dataset --batch-size 16 --workers 0 --phase1-epochs 5 --phase2-epochs 10
```

The script performs two phases for each architecture:

- Phase 1: freeze the feature extractor and train the classification head.
- Phase 2: fine-tune the architecture-specific final feature block and classifier.

Model checkpoints are saved under `models/`. Per-model metrics, classification reports, confusion matrices, and training curves are saved under `results/<model_name>/`.

To train one model only, use `--model resnet50`, `--model efficientnet_b0`, `--model mobilenet_v2`, or `--model vgg16`.

If pretrained weights cannot be downloaded, use `--no-pretrained` (this changes the experiment and must be documented).

### 3. Required output checks

Confirm all four `.pth` files exist in `models/`, and all four `results/<model>/metrics.json` files report the same `test_samples` count. Evaluation loaders use `drop_last=False`, so the final partial batch is included.

`results/model_comparison.csv` is generated from measured metrics. Do not manually enter or reuse the old notebook accuracy values as final results.

## Important reproducibility notes

- All four models use PyTorch and the same 38-class mapping from `class_indices.json`.
- The canonical input size is 224 × 224 RGB with ImageNet normalization.
- Augmentation is applied only to training data.
- Test data must not be used for checkpoint selection or hyperparameter tuning.
- Old notebook scores are historical until the clean split is retrained and evaluated.
- The ZIP originally supplied did not include the PlantVillage dataset or final trained checkpoints; training must be run locally after downloading the dataset.
