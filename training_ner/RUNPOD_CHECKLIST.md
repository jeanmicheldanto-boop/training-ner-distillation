# Checklist de déploiement RunPod

## ✅ Pré-déploiement (Local)

### 1. Validation setup
- [ ] `python validate_setup.py` passe tous les tests
- [ ] Config YAML valide et complète
- [ ] Données JSONL bien formatées
- [ ] Labels cohérents (label2id.json)
- [ ] Dependencies listées dans requirements.txt

### 2. Vérification données
- [ ] train.jsonl : minimum 1000 exemples
- [ ] val.jsonl : minimum 100 exemples  
- [ ] test.jsonl : minimum 100 exemples
- [ ] Tous les labels présents dans label2id.json
- [ ] Pas de tokens vides ou manquants

### 3. Backup
- [ ] Commit git de tout le code
- [ ] Sauvegarder données localement
- [ ] Noter la config exacte utilisée

## 🚀 Déploiement RunPod

### 1. Création instance
- [ ] GPU sélectionné : RTX 4090 / A100
- [ ] Template : PyTorch 2.0+ CUDA 11.8+
- [ ] Volume persistant : 50 GB minimum
- [ ] Ports : SSH (22), Jupyter (8888)

### 2. Upload code et données
```bash
# Option 1: Git clone
git clone <your_repo> /workspace/training_ner

# Option 2: SCP
scp -r training_ner/ root@<runpod_ip>:/workspace/
```

- [ ] Code uploadé dans `/workspace/training_ner`
- [ ] Données dans `/workspace/training_ner/data/`
- [ ] Config dans `/workspace/training_ner/configs/`

### 3. Installation dépendances
```bash
cd /workspace/training_ner
pip install -r requirements.txt
```

- [ ] Toutes les dépendances installées
- [ ] CUDA disponible (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Transformers installé

### 4. Test rapide
```bash
# Valider setup
python validate_setup.py

# Test teacher loading (rapide)
python -c "from models import load_teacher; from utils import load_config; teacher, tok = load_teacher(load_config('configs/kd_camembert.yaml'))"
```

- [ ] Validation passe
- [ ] Teacher se charge sans erreur

## 🎯 Entraînement

### 1. Lancer distillation
```bash
# Screen ou tmux pour éviter déconnexion
screen -S training
python train_kd.py --config configs/kd_camembert.yaml --output /workspace/artifacts/student_10L
```

- [ ] Training lancé
- [ ] Logs visibles
- [ ] GPU utilisé (vérifier avec `nvidia-smi`)

### 2. Monitoring
```bash
# Dans un autre terminal
watch -n 30 nvidia-smi
tail -f /workspace/artifacts/student_10L/training_log.jsonl
```

- [ ] GPU memory stable (pas de OOM)
- [ ] Loss diminue
- [ ] Temps par epoch raisonnable (~30-60 min)

### 3. Checkpoints
- [ ] Checkpoints sauvegardés régulièrement
- [ ] Validation loss logged
- [ ] Pas d'erreurs dans les logs

## 🔪 Pruning

### 1. Après distillation complète
```bash
python prune_heads.py \
  --model /workspace/artifacts/student_10L \
  --rate 0.25 \
  --output /workspace/artifacts/student_10L_pruned
```

- [ ] Pruning terminé
- [ ] Masque sauvegardé (heads_pruned_mask.json)

### 2. Fine-tuning post-pruning
```bash
python finetune_postprune.py \
  --model /workspace/artifacts/student_10L_pruned \
  --output /workspace/artifacts/student_10L_final \
  --epochs 2
```

- [ ] Fine-tuning terminé
- [ ] Modèle final sauvegardé

## 🔍 Test et validation

### 1. Inférence test
```bash
# Créer fichier de test
echo "Jean Dupont habite à Paris." > /workspace/test.txt
echo "Le Dr Martin travaille à Lyon." >> /workspace/test.txt

# Inférence
python inference.py \
  --model /workspace/artifacts/student_10L_final \
  --input /workspace/test.txt \
  --output /workspace/results.jsonl
  
# Vérifier résultats
cat /workspace/results.jsonl
```

- [ ] Inférence fonctionne
- [ ] Entités détectées cohérentes
- [ ] Format JSONL correct

### 2. Évaluation (optionnel)
```bash
# TODO: Script d'évaluation à créer
python evaluate.py \
  --model /workspace/artifacts/student_10L_final \
  --test_file data/test.jsonl
```

- [ ] F1-score calculé
- [ ] Performances acceptables (>85%)

## 📥 Récupération modèle

### 1. Download depuis RunPod
```bash
# Depuis votre machine locale
scp -r root@<runpod_ip>:/workspace/artifacts/student_10L_final ./models/
```

- [ ] Modèle téléchargé
- [ ] Tous les fichiers présents (pytorch_model.bin, config.json, tokenizer)

### 2. Test local
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("./models/student_10L_final")
tokenizer = AutoTokenizer.from_pretrained("./models/student_10L_final")

# Test
text = "Jean habite à Paris"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.shape)
```

- [ ] Modèle charge localement
- [ ] Inférence fonctionne

## 🧹 Cleanup RunPod

### 1. Sauvegarder artefacts importants
- [ ] Modèle final téléchargé
- [ ] Logs de training sauvegardés
- [ ] Masque de pruning sauvegardé
- [ ] Config finale sauvegardée

### 2. Arrêter instance
- [ ] Stop pod (si volume persistant)
- [ ] Terminate pod (si plus besoin)

## 📊 Documentation résultats

### Métriques à noter
- [ ] Temps total d'entraînement
- [ ] Loss finale (train/val)
- [ ] F1-score final (si évalué)
- [ ] Taille modèle (paramètres)
- [ ] Compression ratio vs teacher
- [ ] Vitesse inférence (tokens/sec)

### Problèmes rencontrés
- [ ] Documenter erreurs rencontrées
- [ ] Solutions appliquées
- [ ] Hyperparamètres ajustés

## 🎯 Next steps

- [ ] Déployer modèle en production
- [ ] Créer API REST pour inférence
- [ ] Monitorer performances en production
- [ ] Collecter feedback utilisateurs
- [ ] Itérer sur nouvelles données

---

**Date de déploiement**: ___________
**Durée totale**: ___________
**Coût RunPod**: ___________
**Résultats**: ✅ / ⚠️ / ❌
