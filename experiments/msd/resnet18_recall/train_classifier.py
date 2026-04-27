# script_2_train_resnet_ensemble_recall.py
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import precision_recall_curve # NOUVEL IMPORT

from experiments.msd._shared.proposal_strategy import get_resnet_checkpoint_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrainement Multi-Seed Ensemble (Optimisé Recall) sur : {device}")

checkpoint_dir = get_resnet_checkpoint_dir()
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"Checkpoints ResNet sauvegardes dans : {checkpoint_dir.resolve()}")

BATCH_SIZE = 32
NUM_RUNS = 5
EPOCHS = 20
MIN_PRECISION = 0.80
MODEL_BETA = 2.0
WARMUP_EPOCHS = 3
NUM_WORKERS = 2


def set_seed(seed):
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
    return float(recalls[best_idx]), float(precisions[best_idx]), float(f_beta[best_idx]), float(threshold)

# 1. Transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), 
    transforms.RandomAutocontrast(p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 2. Chargement Datasets
train_dir = "data/classifier_dataset_resnet18/train"
val_dir = "data/classifier_dataset_resnet18/val"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

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

# --- GESTION DU DÉSÉQUILIBRE ---
targets = train_dataset.targets
class_counts = np.bincount(targets)
total_samples = len(targets)
if len(class_counts) < 2 or min(class_counts) == 0:
    raise RuntimeError(f"Dataset invalide: classes trouvees = {class_counts}")

# Poids temperes: on compense le desequilibre sans pousser le modele a accepter
# trop facilement la classe tumeur, ce qui etait la source principale des FP.
weights = np.sqrt(total_samples / (2.0 * class_counts))
class_weights = torch.FloatTensor(weights).to(device)

print(f"Classes train: negatives={class_counts[0]} positives={class_counts[1]}")
print(f"Poids loss temperes: classe 0={weights[0]:.3f} classe 1={weights[1]:.3f}")
print(f"Selection modele: F{MODEL_BETA:.1f} avec precision >= {MIN_PRECISION:.0%}")

ensemble_results = []

for run in range(1, NUM_RUNS + 1):
    set_seed(1000 + run)
    print("\n" + "="*40)
    print(f"LANCEMENT DU MODELE {run}/{NUM_RUNS}")
    print("="*40)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

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
                
                # Récupération des probabilités pour la classe 1 (Tumeur)
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
                
                torch.save(model.state_dict(), checkpoint_dir / f"resnet18_recall_fold_{run}.pth")
                
                with open(checkpoint_dir / f"threshold_resnet18_run_{run}.txt", "w") as f:
                    f.write(str(best_thresh))
                    
                print("  -> Nouveau meilleur modele sauvegarde (meilleur compromis recall/precision) !")

    ensemble_results.append(
        {
            'run': run,
            'recall': best_recall,
            'precision': best_precision,
            'f_beta': best_fbeta,
            'thresh': best_thresh,
        }
    )
    print(
        f"-> Fin du Modele {run}. F{MODEL_BETA:.1f}: {best_fbeta:.4f} | "
        f"Recall: {best_recall:.4f} | Precision: {best_precision:.4f} "
        f"(Seuil local: {best_thresh:.4f})"
    )

print("\n" + "="*40)
print("BILAN DE L'ENSEMBLE (5 MODELES INDEPENDANTS - OPTIMISÉS RECALL)")
print("="*40)
avg_recall = np.mean([r['recall'] for r in ensemble_results])
avg_precision = np.mean([r['precision'] for r in ensemble_results])
avg_fbeta = np.mean([r['f_beta'] for r in ensemble_results])
for r in ensemble_results:
    print(
        f"Modele {r['run']} : F{MODEL_BETA:.1f} = {r['f_beta']:.4f} | "
        f"Recall = {r['recall']:.4f} | Precision = {r['precision']:.4f} | "
        f"Seuil propre = {r['thresh']:.4f}"
    )
print("-" * 40)
print(f"RECALL MOYEN EN VALIDATION : {avg_recall:.4f}")
print(f"PRECISION MOYENNE EN VALIDATION : {avg_precision:.4f}")
print(f"F{MODEL_BETA:.1f} MOYEN EN VALIDATION : {avg_fbeta:.4f}")
