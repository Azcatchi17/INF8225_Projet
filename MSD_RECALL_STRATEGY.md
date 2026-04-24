# Strategie recall MSD

Objectif : reduire les faux negatifs sans laisser repartir les faux positifs.

Le changement principal est de remplacer la decision fragile `top-1 DINO -> ResNet -> MedSAM` par :

1. DINO genere jusqu'a 5 candidats tumeur a bas seuil (`0.01`).
2. Le filtre pancreas est relache mais conserve (`marge 35 px`, overlap minimal `0.05`).
3. ResNet score chaque candidat, puis la decision image-level utilise le meilleur score candidat.
4. MedSAM segmente jusqu'a 2 candidats valides au lieu d'un seul.
5. Le seuil ResNet est recalibre sur validation avec un budget de FP.

Ordre recommande :

```bash
python calibrate_dino.py
python extract_hard_negatives.py
python train_resnet_kfold_recall.py
python calibrate_threshold.py
python test_gd_msd_final_recall.py
```

Fichiers de sortie utiles :

- `optimal_threshold.txt` : seuil ResNet a utiliser en test.
- `optimal_threshold.json` : seuil, metriques validation et configuration de propositions.
- `data/results/calibration_threshold_multi_candidate.csv` : diagnostic validation.
- `data/results/dice_final_report_msd_recall.csv` : diagnostic test final, avec cause des FN.

Dans un notebook, tu peux lancer le test final avec :

```python
%run test_gd_msd_final_recall.py
```

Interpretation des causes de FN dans le CSV final :

- `dino_or_spatial_filters` : aucun candidat n'a survecu a DINO + filtre pancreas.
- `resnet_threshold` : au moins un candidat existe, mais ResNet le rejette.
- `medsam_or_wrong_box` : un candidat est accepte, mais MedSAM ne recouvre pas la tumeur.
