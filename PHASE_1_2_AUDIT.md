# Phase 1 & 2 implementation audit

## Phase 1 changes

1. Replaced filename-based splitting with decoded-RGB SHA-256 grouping before split assignment.
2. Deduplicated repeated images within the same class.
3. Kept cross-class duplicate groups inside one split and made the audit fail until conflicting labels are manually reviewed.
4. Added a group-aware, class-balanced 70/15/15 split assignment.
5. Added `dataset_manifest.jsonl` and regenerated `class_indices.json`/`config.py` in the same sorted order used by `torchvision.datasets.ImageFolder`.
6. Added `audit_dataset.py` to verify class order, mapping, empty classes, image dimensions, duplicate split leakage, and cross-class duplicate conflicts.
7. Added `dataset_exploration.py` for class-distribution and sample-image figures.
8. Updated `dataset_loader.py` so validation and test loaders never drop the final partial batch.
9. Moved old preprocessing outputs to `preprocessing_report_legacy.md` and `preprocessing_metrics_legacy.json`; the old split had five cross-split exact-content hash overlaps. The new canonical report is marked pending until the source dataset is rerun.

## Phase 2 changes

1. Added `train_models.py`, the canonical two-phase PyTorch trainer for ResNet50, EfficientNet-B0, MobileNetV2 and VGG16.
2. Preserved the architecture/head structures in the existing notebooks, including the compact VGG16 head.
3. Phase 1 freezes the feature extractor and trains the classifier. Phase 2 fine-tunes the final architecture-specific feature block and classifier.
4. Checkpoint selection uses validation macro-F1. The test set is evaluated after checkpoint selection and includes every sample.
5. Generates checkpoints, metrics, classification reports, confusion matrices, learning curves, and a measured model-comparison CSV/JSON.
6. Converted `train_vgg16.py` into a compatibility entry point to the canonical trainer, removing the prior architecture mismatch.
7. Fixed MobileNetV2 notebook loader configuration so it does not drop partial batches.
8. Updated the comparison/Grad-CAM notebook to load actual canonical metrics and model paths, and to use `class_indices.json` instead of a separately hardcoded class list. Random-weight Grad-CAM fallback was removed.

## Validation performed in this workspace

- Python syntax compilation passed for all root `.py` files.
- `train_models.py --help` imports and runs successfully.
- All notebook JSON files parse successfully after edits.
- All four model architectures instantiate with 38 outputs; phase-1 and phase-2 trainable parameter sets were checked.
- Synthetic 38-class preprocessing test passed with 70/15/15 totals, 224×224 outputs, correct class mapping, and zero cross-split duplicates.
- Synthetic same-class duplicate test passed: duplicate file was removed and the dataset audit returned PASS.
- Synthetic one-model training smoke test completed both phases, saved a checkpoint and generated metrics/report/plots. Its synthetic accuracy is not a research result and was removed from the deliverable.

## Not yet possible in this workspace

The supplied ZIP does not include the PlantVillage source dataset or final trained checkpoints. Therefore the full 38-class real-data preprocessing, four full training runs, and final real-data model comparison have not been executed here. Run the commands in `PHASE_1_2_EXECUTION.md` on the Windows machine containing the dataset. Do not cite legacy scores as final results.
