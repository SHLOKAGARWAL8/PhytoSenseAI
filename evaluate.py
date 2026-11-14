"""Evaluate the existing AgroDetect EfficientNet-B0 checkpoint on dataset/test.

Run from the project root:
    python evaluate.py
Optional:
    python evaluate.py --batch-size 16 --workers 0

This script performs inference only; it does not train or modify model weights.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "efficientnet_b0_best.pth"
TEST_DIR = ROOT / "dataset" / "test"
CLASS_MAP_PATH = ROOT / "class_indices.json"
OUT_DIR = ROOT / "results" / "efficientnet_b0"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names():
    if not CLASS_MAP_PATH.exists():
        raise FileNotFoundError("class_indices.json is required to verify label order.")
    mapping = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("class_indices.json must be a {class_name: index} mapping.")
    return [name for name, idx in sorted(mapping.items(), key=lambda item: item[1])]


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a state dictionary.")
    state = checkpoint
    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            break
    state = {k: v for k, v in state.items() if torch.is_tensor(v)}
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod.", "model."):
            while key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    if not cleaned:
        raise ValueError("No tensor weights found in checkpoint.")
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    expected_names = load_class_names()
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(TEST_DIR, transform=tfm)
    if dataset.classes != expected_names:
        raise ValueError(
            "Class order mismatch between dataset/test and class_indices.json.\n"
            f"dataset/test: {dataset.classes}\nclass_indices.json: {expected_names}\n"
            "Do not interpret metrics until the training label mapping is verified."
        )
    if len(dataset.classes) != 38:
        raise ValueError(f"Expected 38 classes, found {len(dataset.classes)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(DEVICE.type == "cuda"))

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 38)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.to(DEVICE).eval()

    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(DEVICE, non_blocking=True))
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())

    names = dataset.classes
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "EfficientNet-B0", "checkpoint": str(MODEL_PATH.relative_to(ROOT)),
        "test_dataset": str(TEST_DIR.relative_to(ROOT)), "device": str(DEVICE),
        "test_images": len(dataset), "classes": len(names),
        "accuracy": float(acc), "macro_precision": float(p_macro),
        "macro_recall": float(r_macro), "macro_f1": float(f_macro),
        "weighted_precision": float(p_weight), "weighted_recall": float(r_weight),
        "weighted_f1": float(f_weight),
        "class_mapping_verified": True,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=names, zero_division=0),
        encoding="utf-8")
    with (OUT_DIR / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in names:
            row = report[name]
            writer.writerow([name, row["precision"], row["recall"], row["f1-score"], row["support"]])

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
    ax.set_title("EfficientNet-B0 — Test Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    short = [n.replace("___", " / ").replace("_", " ") for n in names]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    fig.tight_layout(); fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=220); plt.close(fig)

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("EfficientNet-B0 — Normalized Test Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(short, rotation=90, fontsize=6); ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    fig.tight_layout(); fig.savefig(OUT_DIR / "confusion_matrix_normalized.png", dpi=220); plt.close(fig)

    print("\nAGRODETECT — PHASE 5 EVALUATION")
    print("Model:", summary["model"], "| Device:", DEVICE)
    print("Images:", len(dataset), "| Classes:", len(names))
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1"):
        print(f"{key:20s}: {summary[key] * 100:.2f}%")
    print("\nSaved evaluation artifacts to:", OUT_DIR)


if __name__ == "__main__":
    main()
