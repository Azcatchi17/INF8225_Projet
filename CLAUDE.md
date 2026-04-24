# CLAUDE.md — Projet INF8225, segmentation tumeurs pancréatiques

État du projet et décisions prises pendant les sessions Claude.

---

## 1. Contexte

Pipeline de détection + segmentation de tumeurs pancréatiques sur scans CT (MSD Pancreas).

**Dataset** : `data/MSD_pancreas/`
- 800 train / 100 val / 100 test images PNG 512×512
- Masques avec 3 valeurs : `0` (background), `1` (pancréas), `2` (tumeur)
- Métadonnées COCO dans `train.json` / `val.json` / `test.json`
- 60/40 split tumor-présent / tumor-absent dans test

**Modèles** :
- Grounding DINO fine-tuné sur `("tumor",)` via [tumor_config_v3.py](tumor_config_v3.py) → `work_dirs/tumor_config_v3/best_coco_bbox_mAP_epoch_11.pth`
- MedSAM vit_b → `MedSAM/work_dir/MedSAM/medsam_vit_b.pth`
- Tous les assets lourds sont dans Drive folder `1BAcGyja2SHP3t2OFOleN2cND4QlXb3fW`, symlinkés par [colab/setup.py](colab/setup.py)

## 2. Problèmes diagnostiqués

1. **DINO hallucine** : faux positifs massifs sur scans sans tumeur (spec baseline = 0.17)
2. **Negative prompts DINO inefficaces** : `"no tumor"` ne marche pas, l'encodeur BERT ne gère pas la négation
3. **Pas de discrimination tumor / no-tumor globale** : signal trop local pour un classifieur en amont
4. **MedSAM dépend entièrement des boxes DINO** : mauvaise box → mauvaise segmentation
5. **Config DINO v3 sans négatifs explicites** : [tumor_config_v3.py:30](tumor_config_v3.py#L30) a `num_sample_negative=0` → DINO n'a jamais appris "ne rien activer"

## 3. Architecture — pipeline à 3 portes

```
CT slice
   │
   ├─[G1] Pancreas ROI  ──────────── masque anatomique
   │      TinyUNet 2D entraîné sur label==1 du MSD
   │      → mask_pancreas (dilaté) + bbox
   │
   ├─[G2] DINO filtré  ───────────── candidats restants
   │      detect("tumor") → boxes
   │      Filtre : overlap(box, mask_pancreas) ≥ MIN_PANCREAS_OVERLAP
   │      [optionnel] dual-prompt : score_tumor − α · max(score_distractors)
   │
   └─[G3] Décision no-tumor  ──────
          Si max_score < τ (calibré sur val) → masque vide
          [optionnel] VLM verif Gemini sur crop top-1
          Sinon → MedSAM(box clippée à bbox_pancreas)
```

## 4. Fichiers créés

Tous dans [agentic/dino_gemini_msd/](agentic/dino_gemini_msd/) :

- **[pancreas_roi.py](agentic/dino_gemini_msd/pancreas_roi.py)** — Dataset + TinyUNet (4-level, base=32, ~1.9M params) + `train_pancreas_unet(loss_name, tversky_beta, base_channels)` + `infer_pancreas(dilation_px)` avec cache module-level.
- **[gating.py](agentic/dino_gemini_msd/gating.py)** — `GateResult` dataclass + `infer_gated(use_pancreas_roi, use_vlm_verify, use_dual_prompt, score_threshold)` orchestrant G1/G2/G3. Helpers `filter_boxes_by_pancreas`, `differential_score`, `_vlm_verify_crop` (Gemini).
- **[calibrate.py](agentic/dino_gemini_msd/calibrate.py)** — `collect_scores` → `sweep_thresholds` → `CalibResult` avec courbes F1/sens/spec, τ optimal par F1.
- **[eval_improved.py](agentic/dino_gemini_msd/eval_improved.py)** — `run_baseline`, `run_gated`, `run_ablation` retournant `(DataFrame, Summary)` avec sensitivity/specificity/precision/F1/DICE séparés tumor-présent.
- **[colab/improved_pipeline.ipynb](colab/improved_pipeline.ipynb)** — 24 cellules : bootstrap Colab + setup, training UNet, calibration, démo visuelle, ablation 6 configs, confusion matrices, visualisation delta-max.

## 5. Fichiers modifiés

- **[agentic/dino_gemini_msd/config.py](agentic/dino_gemini_msd/config.py)** — section Phase 3 :
  - `PANCREAS_CKPT = work_dirs/pancreas_unet/best.pt`
  - `PANCREAS_DILATION_PX = 35` (v1: 15)
  - `MIN_PANCREAS_OVERLAP = 0.05` (v1: 0.10)
  - `MIN_CENTER_IN_MASK = False` (v1: True)
  - `CLIP_MASK_TO_PANCREAS_ROI = False` (v1: True)
  - `TUMOR_DIFF_THRESHOLD = 0.25` (calibré runtime)
  - `DUAL_PROMPT_DISTRACTORS = ["normal pancreas tissue", "bowel gas", "abdominal organ"]`
  - `DUAL_PROMPT_ALPHA = 0.5`

- **[agentic/dino_gemini_msd/agent.py](agentic/dino_gemini_msd/agent.py)** — `run_agent(..., use_gating=True, gating_kwargs=...)` : appelle `infer_gated` avant la boucle itérative ; rejet → retour d'état vide avec `stop_reason="gated:..."` ; sinon la box gatée est promue en tête de liste de candidats.

- **[colab/setup.py](colab/setup.py)** — ajout de `work_dirs/pancreas_unet` et `data/agent_results_msd/improved` dans `OUTPUT_SUBDIRS` (créés sur Drive).

## 6. Itérations — résultats chiffrés

Split test = 100 images (60 tumor-présent / 40 no-tumor).

| Version | UNet (val DICE pancréas) | Sens | Spec | DICE tumor | F1 | TP/FP/TN/FN |
|---|---|---|---|---|---|---|
| Baseline DINO+MedSAM | — | 0.98 | 0.17 | 0.47 | 0.78 | 59/33/7/1 |
| V1 gated (strict) | 256px, 15ep, BCE+Dice → **0.41** | 0.73 | 0.50 | 0.34 | 0.71 | 44/20/20/16 |
| V2 gated (relaxed) | 384px, 30ep, Tversky β=0.7 → **~0.60** | 0.92 | 0.30 | 0.45 | 0.77 | 55/28/12/5 |

**Leçons** :
- V1 trop strict : gagne en spec (0.17→0.50) mais casse tout le reste (sens ↓ 25pts, DICE ↓ 13pts)
- V2 : UNet Tversky + filtres relâchés → sens récupérée, DICE revenu, mais **FP remontés** (20→28) parce que le ROI est plus généreux et laisse passer des hallucinations intra-pancréas
- V3 (en cours) : ajout de dual-prompt + VLM verif dans l'ablation pour tenter de filtrer les FP résiduels

## 7. État actuel (2026-04-23)

**Poussé sur `moradBMH:temp` et `Azcatchi17:temp`** (force-push) :
- `2764fa4` rétabli nom de chemin
- `db86c2c` test avec gemini et dual prompt (ablation 6 configs)
- `0ece887` v2 unet (Tversky, 384px, 30ep)

**En attente** : exécution du notebook Colab avec les 6 configs d'ablation pour voir l'impact marginal de dual-prompt et VLM.

## 8. Prochaines étapes (par ordre d'impact attendu)

### Court terme (runtime, sans retrain lourd)

1. **Lancer l'ablation 6 configs** dans `improved_pipeline.ipynb` (section 6) — analyser :
   - `+G1+G3+dual_prompt` — attendu : FP −10 à 20%
   - `+G1+G3+VLM` — attendu : FP −40%
   - `+G1+G3+dual+VLM` — attendu : FP −50%
   - Tradeoff : chaque couche supprimante peut coûter 1-2 TP

2. **Activer le run_agent avec gating** pour pousser le DICE au-delà de 0.50 (raffinement itératif Gemini sur la box gatée).

### Moyen terme (retrain DINO, impact racine)

3. **Fine-tuning DINO v4** avec négatifs explicites — vrai correctif des hallucinations :
   - Créer `tumor_config_v4.py` avec `num_sample_negative=3` (vs 0 en v3)
   - Vérifier que `train.json` inclut bien les scans sans tumeur (annotations vides)
   - Retrain depuis le checkpoint pré-entraîné (~30 min T4)
   - Attendu : FP 28 → ~14, DICE 0.45 → ~0.52

4. **Contraste / windowing CT** — vérifier si les PNG sont bien fenêtrés pour le pancréas (W=400, L=40 HU). Si non, re-pré-traiter le dataset. Impact : +5 pts DICE potentiels.

### Long terme (pipeline complet)

5. **Multi-slice context** — les tumeurs apparaissent sur plusieurs coupes adjacentes, les hallucinations non. Agréger la décision sur 3 coupes consécutives → specificity boostée sans coût sensitivity.

6. **BiomedCLIP pour la vérification** — remplacer Gemini par BiomedCLIP sur le crop (zero-shot, embeddings médicaux) → plus rapide, plus spécifique, pas de rate-limit.

## 9. Commandes utiles

```bash
# Re-train UNet proprement (supprime le .pt existant d'abord)
rm work_dirs/pancreas_unet/best.pt
# Puis dans le notebook : RETRAIN_UNET = True

# Sync local → remote
git push origin temp
git push upstream temp          # si tu as les droits Azcatchi17

# État git propre après rebase raté
git reflog                       # trouve ton commit
git reset --hard <sha>           # restaure
git push --force-with-lease origin temp
```

## 10. Drive layout

Projet pointe vers `/content/drive/MyDrive/Projet_Medsam/` en Colab :

```
Projet_Medsam/
├── data/
│   ├── MSD_pancreas/            # dataset (train/val/test + *.json)
│   ├── agent_results_msd/       # calibration.json, ablation_*.csv
│   └── agent_results_improved/  # résultats notebook V2
├── work_dirs/
│   ├── tumor_config_v3/         # DINO fine-tuné
│   └── pancreas_unet/best.pt    # UNet pancréas entraîné
├── work_dir/                    # MedSAM checkpoints
└── grounding_dino_swin-t_pretrain_obj365_goldg_*.pth
```
