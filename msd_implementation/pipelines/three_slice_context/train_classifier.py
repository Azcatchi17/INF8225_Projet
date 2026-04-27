"""3-slice variant of train_resnet_kfold_recall.py.

Trains an ensemble of 5 ResNet-18 on the 3-slice patch dataset
(``data/classifier_dataset_three_slice/``). Saves checkpoints with the
``three_slice_fold_*.pth`` prefix so they coexist with the original
2D ensemble.

Usage (from repo root):
    python -m msd_implementation.pipelines.three_slice_context.train_classifier
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msd_implementation.pipelines.common.proposal_strategy import get_resnet_checkpoint_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrainement Multi-Seed Ensemble 3-slice sur : {device}")

checkpoint_dir = get_resnet_checkpoint_dir()
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"Checkpoints ResNet 3-slice sauvegardes dans : {checkpoint_dir.resolve()}")

BATCH_SIZE = 32
NUM_RUNS = 5
EPOCHS = 20
MIN_PRECISION = 0.80
MODEL_BETA = 2.0
WARMUP_EPOCHS = 3
NUM_WORKERS = 2

CHECKPOINT_PREFIX = "three_slice_fold"
THRESHOLD_PREFIX = "threshold_three_slice_run"

TRAIN_DIR = "data/classifier_dataset_three_slice/train"
VAL_DIR = "data/classifier_dataset_three_slice/val"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def best_pr_point(labels, probs, min_precision=MIN_PRECISION, beta=MODEL_BETA):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    valid = np.where(precisions >= min_precision)[0]
    if len(valid) == 0:
        valid = np.arange(len(precisions))

    beta2 = beta * beta
    f_beta = np.zeros_like(precisions)
    denom = beta2 * precisions + recalls
    ok = denom > 0
    f_beta[ok] = (1 + beta2) * precisions[ok] * recalls[ok] / denom[ok]

    best_idx = valid[np.argmax(f_beta[valid])]
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    return (
        float(recalls[best_idx]),
        float(precisions[best_idx]),
        float(f_beta[best_idx]),
        float(threshold),
    )


# Note: the patches are stored as RGB PNGs whose channels carry
# [prev_slice, curr_slice, next_slice]. We keep the same Normalize stats
# as the 2D pipeline (`[0.5,0.5,0.5]`) since the per-channel intensity
# distribution is similar (each channel is still a CT-window grayscale).
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

targets = train_dataset.targets
class_counts = np.bincount(targets)
total_samples = len(targets)
if len(class_counts) < 2 or min(class_counts) == 0:
    raise RuntimeError(f"Dataset invalide: classes trouvees = {class_counts}")

weights = np.sqrt(total_samples / (2.0 * class_counts))
class_weights = torch.FloatTensor(weights).to(device)

print(f"Classes train: negatives={class_counts[0]} positives={class_counts[1]}")
print(f"Poids loss temperes: classe 0={weights[0]:.3f} classe 1={weights[1]:.3f}")
print(f"Selection modele: F{MODEL_BETA:.1f} avec precision >= {MIN_PRECISION:.0%}")

ensemble_results = []

for run in range(1, NUM_RUNS + 1):
    set_seed(1000 + run)
    print("\n" + "=" * 40)
    print(f"LANCEMENT DU MODELE 3-SLICE {run}/{NUM_RUNS}")
    print("=" * 40)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_recall = -1.0
    best_precision = -1.0
    best_fbeta = -1.0
    best_thresh = 0.5

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_labels_list = []
        all_probs_list = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs_list.extend(probs.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        epoch_recall, epoch_prec, epoch_fbeta, epoch_thresh = best_pr_point(
            all_labels_list,
            all_probs_list,
            min_precision=MIN_PRECISION,
            beta=MODEL_BETA,
        )

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | F{MODEL_BETA:.1f}: {epoch_fbeta:.4f} "
            f"| Recall: {epoch_recall:.4f} (Prec: {epoch_prec:.2f} @ tau={epoch_thresh:.2f})"
        )

        if epoch >= WARMUP_EPOCHS:
            is_better = (
                epoch_fbeta > best_fbeta
                or (epoch_fbeta == best_fbeta and epoch_recall > best_recall)
                or (
                    epoch_fbeta == best_fbeta
                    and epoch_recall == best_recall
                    and epoch_prec > best_precision
                )
            )

            if is_better:
                best_recall = epoch_recall
                best_precision = epoch_prec
                best_fbeta = epoch_fbeta
                best_thresh = epoch_thresh

                torch.save(
                    model.state_dict(),
                    checkpoint_dir / f"{CHECKPOINT_PREFIX}_{run}.pth",
                )

                with open(checkpoint_dir / f"{THRESHOLD_PREFIX}_{run}.txt", "w") as f:
                    f.write(str(best_thresh))

                print(
                    "  -> Nouveau meilleur modele 3-slice sauvegarde !"
                )

    ensemble_results.append(
        {
            "run": run,
            "recall": best_recall,
            "precision": best_precision,
            "f_beta": best_fbeta,
            "thresh": best_thresh,
        }
    )
    print(
        f"-> Fin du Modele 3-slice {run}. F{MODEL_BETA:.1f}: {best_fbeta:.4f} | "
        f"Recall: {best_recall:.4f} | Precision: {best_precision:.4f} "
        f"(Seuil local: {best_thresh:.4f})"
    )

print("\n" + "=" * 40)
print("BILAN DE L'ENSEMBLE 3-SLICE (5 MODELES)")
print("=" * 40)
avg_recall = np.mean([r["recall"] for r in ensemble_results])
avg_precision = np.mean([r["precision"] for r in ensemble_results])
avg_fbeta = np.mean([r["f_beta"] for r in ensemble_results])
for r in ensemble_results:
    print(
        f"Modele {r['run']} : F{MODEL_BETA:.1f} = {r['f_beta']:.4f} | "
        f"Recall = {r['recall']:.4f} | Precision = {r['precision']:.4f} | "
        f"Seuil propre = {r['thresh']:.4f}"
    )
print("-" * 40)
print(f"RECALL MOYEN VAL : {avg_recall:.4f}")
print(f"PRECISION MOYENNE VAL : {avg_precision:.4f}")
print(f"F{MODEL_BETA:.1f} MOYEN VAL : {avg_fbeta:.4f}")
