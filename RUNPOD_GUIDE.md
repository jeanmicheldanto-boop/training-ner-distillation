# 🚀 Guide complet : De local à RunPod

## Phase 1 : Créer un repo GitHub (5 minutes)

### Étape 1 : Créer le repo sur GitHub

1. Allez sur **https://github.com/new**
2. Remplissez :
   - **Repository name** : `ner-distillation` (ou ce que vous voulez)
   - **Description** : "NER distillation pipeline - Knowledge Distillation + Pruning for CamemBERT"
   - **Visibility** : Public (sinon RunPod ne pourra pas cloner sans token)
   - **Cocher** : "Add a README file"
3. Cliquez **"Create repository"**

### Étape 2 : Récupérer l'URL du repo

Après création, vous verrez un bouton vert **"Code"**. Cliquez dessus et copiez l'URL HTTPS :
```
https://github.com/jeanmicheldanto-boop/ner-distillation.git
```

### Étape 3 : Ajouter le remote et pousser

```bash
cd c:\Users\Lenovo\dataner
git remote add origin https://github.com/jeanmicheldanto-boop/ner-distillation.git
git branch -M main
git push -u origin main
```

**Note** : Il vous demandera votre username/password. Utilisez un **Personal Access Token** à la place du password :
1. Allez sur **https://github.com/settings/tokens**
2. Cliquez **"Generate new token"** → **"Generate new token (classic)"**
3. Donnez-lui un nom : `runpod-access`
4. Cochez : `repo`, `admin:repo_hook`
5. Générez et **copiez le token** (ne le perdez pas !)
6. Quand git demande le password, collez ce token

---

## Phase 2 : Configurer l'Instance RunPod (10 minutes)

### Étape 1 : Se connecter à RunPod

1. Allez sur **https://www.runpod.io**
2. Connectez-vous avec votre compte
3. Allez à **"Pods"** → **"Create New"**

### Étape 2 : Sélectionner le GPU et le Template

**Templates** :
- Cherchez **"PyTorch"** ou **"CUDA 11.8"**
- Sélectionnez une image récente (2024+)

**GPU Selection** :
- **RTX 4090** (24 GB) : ~$0.70/h - BON CHOIX
- **A100 40GB** : ~$1.50/h - EXCELLENT mais 2x plus cher
- **L40S** : ~$0.50/h - OK mais risque VRAM

**Je recommande RTX 4090** pour débuter.

### Étape 3 : Configuration du Pod

| Option | Valeur |
|--------|--------|
| GPU Count | 1 |
| Volume | 50 GB (pour datasets + checkpoints) |
| Container Disk | 20 GB |
| Expose HTTP Ports | 8888 (Jupyter, optionnel) |

Cliquez **"Deploy"** et attendez 1-2 minutes.

### Étape 4 : Accéder à votre Pod

Une fois "Running" :
- Cliquez sur le Pod
- Cliquez **"Connect"** ou **"SSH"**
- Copiez la commande SSH

---

## Phase 3 : Sur RunPod (SSH dans le Pod)

```bash
# Se connecter (remplacer par votre SSH command)
ssh -p YOUR_PORT root@YOUR_IP
```

### Étape 1 : Cloner le repo

```bash
cd /workspace
git clone https://github.com/jeanmicheldanto-boop/ner-distillation.git
cd ner-distillation
```

### Étape 2 : Installer les dépendances

```bash
# Mise à jour système (optionnel mais recommandé)
apt-get update && apt-get upgrade -y

# Installer dépendances Python
pip install --upgrade pip
pip install -r training_ner/requirements.txt

# Vérifier CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Durée estimée** : 5-10 minutes

### Étape 3 : Uploader vos données

**Option A : Via SCP (direct depuis votre machine)**

```bash
# DEPUIS VOTRE PC (local) :
scp -P YOUR_PORT c:\Users\Lenovo\dataner\corpus\corpus_fr_100k_medico_FINAL.txt root@YOUR_IP:/workspace/ner-distillation/data/
```

**Option B : Via le volume persistant RunPod**

Si vous avez un volume persistant, uploadez via le dashboard RunPod.

### Étape 4 : Préparer les données

```bash
cd /workspace/ner-distillation
python training_ner/annotate_corpus.py \
  --input ./data/corpus_fr_100k_medico_FINAL.txt \
  --output ./training_ner/data/ \
  --split 0.8 0.1 0.1 \
  --max_samples 96000 \
  --teacher Jean-Baptiste/camembert-ner

# Vérifier
ls -lah training_ner/data/
```

**Durée estimée** : 2-4 heures (l'annotation auto est long)

### Étape 5 : Valider le setup

```bash
cd training_ner
python validate_setup.py
```

Si tout est ✅ : vous êtes prêt !

---

## Phase 4 : Lancer le Training

### Étape 1 : Distillation

```bash
cd /workspace/ner-distillation/training_ner

# Lancer dans un screen/tmux pour survivre aux déconnexions
screen -S training
python train_kd.py \
  --config configs/kd_camembert.yaml \
  --output artifacts/student_10L

# Pour quitter screen sans l'arrêter : Ctrl+A puis D
# Pour se reconnecter : screen -r training
```

**Durée estimée** : 4-8 heures (10 epochs)

**Monitoring** :
```bash
# Dans un autre SSH terminal :
tail -f /workspace/ner-distillation/training_ner/artifacts/student_10L/training_log.jsonl
watch -n 10 nvidia-smi  # Voir GPU usage
```

### Étape 2 : Pruning

```bash
python prune_heads.py \
  --model artifacts/student_10L \
  --rate 0.25 \
  --output artifacts/student_10L_pruned
```

**Durée estimée** : 30-60 min

### Étape 3 : Fine-tuning post-pruning

```bash
python finetune_postprune.py \
  --model artifacts/student_10L_pruned \
  --output artifacts/student_10L_final \
  --epochs 2
```

**Durée estimée** : 1-2 heures

---

## Phase 5 : Récupérer les Résultats

### Télécharger le modèle final

```bash
# DEPUIS VOTRE PC (local) :
scp -rP YOUR_PORT root@YOUR_IP:/workspace/ner-distillation/training_ner/artifacts/student_10L_final ./my_student_model/
```

### Arrêter le Pod

```bash
# Dans le dashboard RunPod
# Cliquez "Terminate" (arrête et facture l'heure en cours)
# ou "Stop" si vous voulez le reprendre
```

---

## 💰 Estimation des Coûts

| Étape | Durée | Coût (RTX 4090 @$0.70/h) |
|-------|-------|--------------------------|
| Annotation auto | 2-4h | $1.40-2.80 |
| Distillation | 4-8h | $2.80-5.60 |
| Pruning | 0.5-1h | $0.35-0.70 |
| Fine-tuning | 1-2h | $0.70-1.40 |
| **TOTAL** | **7.5-15h** | **$5.25-10.50** |

**Très raisonnable pour un modèle optimisé !**

---

## ⚠️ Troubleshooting

### "CUDA out of memory"
```bash
# Réduire batch_size dans configs/kd_camembert.yaml
# batch_size: 16 → 8 ou 4
```

### "Teacher model not found"
```bash
# Vérifier connexion internet sur RunPod
# pip install transformers --upgrade
```

### "Déconnexion SSH perdra mon entraînement"
```bash
# Utilisez screen ou tmux :
screen -S train
python train_kd.py ...
# Ctrl+A, D pour détacher
# screen -r train pour revenir
```

### "Données too large"
```bash
# Utiliser le volume persistant RunPod ou réduire --max_samples
# python annotate_corpus.py ... --max_samples 50000
```

---

## Prochaines étapes

1. ✅ Créer repo GitHub et pousser le code
2. ✅ Créer pod RunPod
3. ✅ Cloner, installer, valider
4. ✅ Uploader corpus
5. ✅ Annoter + Distiller + Pruner
6. ✅ Télécharger modèle final
7. ✅ Tester dans votre pipeline production
8. ✅ **Si F1 pipeline >= -1.0 → GO** sinon NO-GO

Besoin d'aide à une étape ? 🚀
