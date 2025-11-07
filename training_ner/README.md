# Training NER - Knowledge Distillation Pipeline

Pipeline complet de distillation et pruning pour modèle NER français médico-social.

## 🎯 Objectif

Créer un modèle student léger (10 couches, 25% têtes prunées) à partir du teacher `Jean-Baptiste/camembert-ner` pour déploiement sur GPU RunPod.

## 📁 Structure

```
training_ner/
├── configs/
│   └── kd_camembert.yaml          # Configuration complète
├── data/
│   ├── train.jsonl                # Données d'entraînement
│   ├── val.jsonl                  # Données de validation
│   ├── test.jsonl                 # Données de test
│   └── label2id.json              # Mapping labels NER
├── artifacts/                      # Modèles sauvegardés
│   ├── student_10L/               # Student après distillation
│   ├── student_10L_pruned/        # Student après pruning
│   └── student_10L_final/         # Student après fine-tuning
├── models.py                       # Définition Teacher/Student
├── losses.py                       # Pertes de distillation
├── pruning.py                      # Pruning têtes d'attention
├── data_loader.py                  # Chargement données
├── utils.py                        # Utilitaires
├── train_kd.py                     # Script distillation
├── prune_heads.py                  # Script pruning
├── finetune_postprune.py          # Script fine-tuning
└── inference.py                    # Script inférence
```

## 📋 Format des données

### JSONL Format (train.jsonl, val.jsonl, test.jsonl)
```json
{"tokens": ["Jean", "habite", "à", "Paris"], "ner_tags": ["B-PER", "O", "O", "B-LOC"]}
{"tokens": ["Le", "Dr", "Martin", "travaille"], "ner_tags": ["O", "O", "B-PER", "O"]}
```

### label2id.json
```json
{
  "O": 0,
  "B-PER": 1,
  "I-PER": 2,
  "B-LOC": 3,
  "I-LOC": 4,
  "B-ORG": 5,
  "I-ORG": 6
}
```

## 🚀 Pipeline d'utilisation

### 1. Distillation (Knowledge Distillation)

Entraîner le student avec le teacher comme référence.

```bash
python train_kd.py \
  --config configs/kd_camembert.yaml \
  --output artifacts/student_10L
```

**Durée estimée**: 4-8 heures (10 epochs sur GPU A100)

**Sorties**:
- `artifacts/student_10L/pytorch_model.bin`
- `artifacts/student_10L/config.json`
- `artifacts/student_10L/tokenizer_config.json`
- `artifacts/student_10L/training_log.jsonl`

### 2. Pruning (25% têtes d'attention)

Pruner les têtes les moins importantes.

```bash
python prune_heads.py \
  --model artifacts/student_10L \
  --rate 0.25 \
  --output artifacts/student_10L_pruned
```

**Durée estimée**: 30-60 minutes

**Sorties**:
- `artifacts/student_10L_pruned/pytorch_model.bin`
- `artifacts/student_10L_pruned/heads_pruned_mask.json`

### 3. Fine-tuning post-pruning

Récupérer les performances après pruning.

```bash
python finetune_postprune.py \
  --model artifacts/student_10L_pruned \
  --output artifacts/student_10L_final \
  --epochs 2
```

**Durée estimée**: 1-2 heures

### 4. Inférence

Extraire entités NER sur nouveaux textes.

```bash
python inference.py \
  --model artifacts/student_10L_final \
  --input phrases.txt \
  --output entities.jsonl
```

**Format sortie** (entities.jsonl):
```json
{"text": "Jean habite à Paris", "entities": [{"text": "Jean", "type": "PER", "start": 0, "end": 1}, {"text": "Paris", "type": "LOC", "start": 3, "end": 4}]}
```

## 🖥️ Configuration RunPod

### 1. Créer instance RunPod

- **GPU**: RTX 4090 (24 GB) ou A100 (40/80 GB)
- **Template**: PyTorch 2.0+ CUDA 11.8
- **Volume**: 50 GB pour datasets et checkpoints

### 2. Installation dépendances

```bash
cd /workspace
git clone <your_repo>
cd training_ner
pip install -r requirements.txt
```

### 3. Préparer données

```bash
# Uploader vos données JSONL
# /workspace/data/train.jsonl
# /workspace/data/val.jsonl
# /workspace/data/test.jsonl
# /workspace/data/label2id.json
```

### 4. Lancer entraînement

```bash
# Adapter paths dans configs/kd_camembert.yaml
python train_kd.py --config configs/kd_camembert.yaml --output /workspace/artifacts/student_10L
```

## ⚙️ Configuration principale (kd_camembert.yaml)

Voir `configs/kd_camembert.yaml` pour configuration complète.

**Paramètres clés**:
- `teacher.model_name`: `Jean-Baptiste/camembert-ner`
- `student.num_layers`: `10` (réduit de 12)
- `distillation.temperature`: `2.5`
- `pruning.rate`: `0.25` (25% têtes prunées)
- `training.batch_size`: `16`
- `training.learning_rate`: `2e-5`
- `training.max_epochs`: `10`

## 📊 Monitoring

Les métriques sont loggées dans:
- `artifacts/student_10L/training_log.jsonl` (par step)
- `artifacts/student_10L/training_summary.json` (résumé)

## 🔍 Validation setup

Avant de lancer sur RunPod, valider localement:

```bash
# Vérifier format données
python -c "from data_loader import verify_data_format; verify_data_format('data/train.jsonl')"

# Vérifier config
python -c "from utils import load_config; print(load_config('configs/kd_camembert.yaml'))"
```

## 📈 Performances attendues

**Modèle teacher** (Jean-Baptiste/camembert-ner):
- Paramètres: ~110M
- F1-score: ~90% (sur corpus français général)

**Modèle student final** (10L + pruning 25%):
- Paramètres: ~60-70M (compression ~1.5-1.8x)
- F1-score: ~85-88% (perte acceptable 2-5%)
- Latence: ~40-50% plus rapide

## ❓ Troubleshooting

### Erreur: CUDA Out of Memory
- Réduire `batch_size` dans config (16 → 8)
- Désactiver `mixed_precision` si problème

### Erreur: Teacher model not found
- Vérifier connexion internet
- Vérifier `teacher.model_name` dans config

### Données mal formatées
- Vérifier format JSONL (une ligne = un JSON)
- Vérifier longueur tokens = longueur ner_tags

## 📚 Références

- **Patient KD**: https://arxiv.org/abs/1908.09355
- **CamemBERT**: https://arxiv.org/abs/1911.03894
- **Pruning Heads**: https://arxiv.org/abs/1905.10650

## 📝 TODO pour RunPod

Les sections marquées `TODO: Implémenter sur RunPod` dans le code nécessitent implémentation complète:
- [ ] Forward passes teacher/student avec extraction hidden states
- [ ] Calcul réel des 4 pertes (CE, KD, Hidden, CRF)
- [ ] Calcul importance têtes d'attention (gradient × activation)
- [ ] Masquage effectif des poids Q/K/V/O des têtes prunées
- [ ] Chargement/sauvegarde checkpoints
- [ ] Extraction entités NER avec alignement subwords

## 📞 Support

Pour questions: [votre contact]
