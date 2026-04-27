# Strategie recall MSD

Objectif : reduire les faux negatifs sans laisser repartir les faux positifs.

Le changement principal est de remplacer la decision fragile `top-1 DINO -> ResNet -> MedSAM` par :

1. DINO genere jusqu'a 5 candidats tumeur au runtime (`0.05`).
2. Le filtre pancreas est relache mais conserve (`marge 35 px`, overlap minimal `0.05`).
3. ResNet score chaque candidat avec un ensemble de 5 modeles, puis la decision image-level utilise le meilleur score candidat.
4. MedSAM segmente jusqu'a 2 candidats valides au lieu d'un seul.
5. Le seuil ResNet est recalibre sur validation avec un budget de FP.

Le dataset ResNet est genere plus agressivement que le runtime :

- DINO mine les hard negatives a `0.01`, `top_k=12`, avec un filtre pancreas tres relache.
- Les positifs GT sont moins sur-augmentes pour eviter un dataset trop positif.
- Le dataset final vise au moins `2:1` negatifs/positif et au plus `4:1`.
- Le score final penalise legerement le desaccord entre les 5 ResNet (`mean - 0.25 * std`).

Ordre recommande :

```bash
python calibrate_dino.py
python extract_hard_negatives.py
python train_resnet_kfold_recall.py
python calibrate_threshold.py
python test_gd_msd_final_recall.py
```

Fichiers de sortie utiles :

- `outputs/msd_implementation/resnet18_recall/metrics/optimal_threshold_resnet18.txt` : seuil ResNet a utiliser en test.
- `outputs/msd_implementation/resnet18_recall/metrics/optimal_threshold_resnet18.json` : seuil, metriques validation et configuration de propositions.
- `outputs/msd_implementation/resnet18_recall/metrics/calibration_threshold_multi_candidate.csv` : diagnostic validation.
- `outputs/msd_implementation/resnet18_recall/metrics/threshold_sweep_multi_candidate.csv` : sweep validation des seuils ResNet.
- `outputs/msd_implementation/resnet18_recall/metrics/dice_final_report_resnet18_recall.csv` : diagnostic test final, avec cause des FN.

Dans un notebook, tu peux lancer le test final avec :

```python
%run test_gd_msd_final_recall.py
```

Interpretation des causes de FN dans le CSV final :

- `dino_or_spatial_filters` : aucun candidat n'a survecu a DINO + filtre pancreas.
- `resnet_threshold` : au moins un candidat existe, mais ResNet le rejette.
- `medsam_or_wrong_box` : un candidat est accepte, mais MedSAM ne recouvre pas la tumeur.
