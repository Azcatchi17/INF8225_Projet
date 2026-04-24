# script_2_train_resnet_ensemble_recall.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import precision_recall_curve # NOUVEL IMPORT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrainement Multi-Seed Ensemble (Optimisé Recall) sur : {device}")

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
train_dir = "data/classifier_dataset_hard/train"
val_dir = "data/classifier_dataset_hard/val"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- GESTION DU DÉSÉQUILIBRE ---
targets = train_dataset.targets
class_counts = np.bincount(targets)
total_samples = len(targets)
weights = [total_samples / class_counts[0], total_samples / class_counts[1]]
class_weights = torch.FloatTensor(weights).to(device)

NUM_RUNS = 5
epochs = 15
MIN_PRECISION = 0.70 # On exige 70% de prédictions correctes parmi les alertes
ensemble_results = []

for run in range(1, NUM_RUNS + 1):
    print("\n" + "="*40)
    print(f"LANCEMENT DU MODELE {run}/{NUM_RUNS}")
    print("="*40)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_recall = -1.0 
    best_prec_at_best_recall = -1.0 # NOUVEAU : Pour le tie-breaker
    best_thresh = 0.5

    for epoch in range(epochs):
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
        
        # ==========================================
        # CALCUL DU RECALL OPTIMAL DE L'ÉPOQUE
        # ==========================================
        precisions, recalls, thresholds = precision_recall_curve(all_labels_list, all_probs_list)
        
        # On cherche le meilleur Recall où la Précision est >= MIN_PRECISION
        valid_mask = precisions >= MIN_PRECISION
        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            # Parmi les valides, on prend l'index avec le recall maximum
            best_idx = valid_indices[np.argmax(recalls[valid_indices])]
            
            epoch_recall = recalls[best_idx]
            epoch_prec = precisions[best_idx]
            epoch_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        else:
            epoch_recall, epoch_prec, epoch_thresh = 0.0, 0.0, 0.5
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Recall: {epoch_recall:.4f} (Prec: {epoch_prec:.2f} @ τ={epoch_thresh:.2f})")
        
        # SAUVEGARDE SUR LE RECALL AVEC TIE-BREAKER (Warm-up > 3)
        if epoch >= 3:
            # Condition 1 : On bat le record absolu de Recall
            is_best_recall = epoch_recall > best_recall
            
            # Condition 2 : Cas d'égalité parfaite du Recall, on regarde la Précision
            is_tie_breaker = (epoch_recall == best_recall) and (epoch_prec > best_prec_at_best_recall)
            
            if is_best_recall or is_tie_breaker:
                best_recall = epoch_recall
                best_prec_at_best_recall = epoch_prec
                best_thresh = epoch_thresh
                
                torch.save(model.state_dict(), f"resnet_fold_{run}.pth")
                
                with open(f"threshold_run_{run}.txt", "w") as f:
                    f.write(str(best_thresh))
                    
                if is_tie_breaker:
                    print("  -> Nouveau meilleur modele sauvegarde (Egalite Recall, Meilleure Precision) !")
                else:
                    print("  -> Nouveau meilleur modele sauvegarde (Meilleur Recall absolu) !")

    ensemble_results.append({'run': run, 'recall': best_recall, 'thresh': best_thresh})
    print(f"-> Fin du Modele {run}. Meilleur Recall : {best_recall:.4f} (Seuil local: {best_thresh:.4f})")

print("\n" + "="*40)
print("BILAN DE L'ENSEMBLE (5 MODELES INDEPENDANTS - OPTIMISÉS RECALL)")
print("="*40)
avg_recall = np.mean([r['recall'] for r in ensemble_results])
for r in ensemble_results:
    print(f"Modele {r['run']} : Recall = {r['recall']:.4f} | Seuil propre = {r['thresh']:.4f}")
print("-" * 40)
print(f"RECALL MOYEN EN VALIDATION : {avg_recall:.4f}")