# ✅ CHECKLIST FINALE AVANT PUSH & TEST POST-BUILD

## 📋 Vérifications Pré-Push (COMPLÉTÉES)

### Code Python
- [x] **handler.py** : Syntaxe OK, import sys ajouté, sys.argv configuré pour annotate et train
- [x] **upload_corpus.py** : Syntaxe OK, gestion erreurs, encode base64 correctement
- [x] **workflow.py** : Syntaxe OK, orchestration annotation→training
- [x] **monitor_jobs.py** : Syntaxe OK, polling avec timeouts
- [x] **test_endpoint.py** : Syntaxe OK, utilise api.runpod.ai (corrigé)

### Configuration Docker
- [x] **Dockerfile** : CMD utilise runpod.serverless.worker
- [x] **training_ner/requirements.txt** : Dépendances listées (torch, transformers, etc.)
- [x] **.dockerignore** : Exclut data_local/, corpus/, .venv/ (build rapide)
- [x] **__init__.py** : Présents dans racine et training_ner/ (imports OK)

### Handler Actions
- [x] **upload_corpus** : Décode base64, crée répertoires, écrit fichier, vérifie
- [x] **annotate** : Configure sys.argv avec --input, --output, --model_name, --batch_size
- [x] **train** : Configure sys.argv avec --config

### Scripts d'Annotation et Training
- [x] **annotate_corpus.py** : main() appelle parse_args() → sys.argv doit être configuré ✅
- [x] **train_kd.py** : main() appelle parse_args() → sys.argv doit être configuré ✅

---

## 🚀 PLAN DE TEST POST-BUILD (30 minutes de build)

### Étape 1: Upload du Corpus (2-3 minutes)
```powershell
# Variables d'environnement déjà définies
$env:RUNPOD_ENDPOINT_ID = "wupg1xsork5mk7"
$env:RUNPOD_API_KEY = "VOTRE_CLE_API_ICI"

# Upload du gros corpus (96k lignes, ~10-20 MB)
python upload_corpus.py corpus\corpus_fr_100k_medico_FINAL.txt /workspace/corpus_fr_100k_medico_FINAL.txt

# Suivre le job d'upload
python monitor_jobs.py "<JOB_ID_AFFICHÉ>"
```

**Résultat attendu :**
- Job complété en < 1 minute
- Output indique : "Corpus uploaded to /workspace/corpus_fr_100k_medico_FINAL.txt"
- Lignes : ~96230
- Taille : ~10-20 MB

**Si échec :**
- Vérifier les logs du job dans monitor_jobs.py
- Vérifier que le chemin /workspace est accessible (ou essayer /app/corpus)

---

### Étape 2: Test Annotation (5-10 minutes avec mini-corpus)
```powershell
# D'abord tester avec le mini-corpus
python upload_corpus.py corpus\corpus_test_100.txt /workspace/corpus_test_100.txt

# Suivre upload
python monitor_jobs.py "<JOB_ID>"

# Lancer annotation du mini-corpus
python test_endpoint.py  # Modif à faire pour pointer vers /workspace/corpus_test_100.txt
```

**OU directement avec workflow.py :**
```powershell
python workflow.py --corpus-path /workspace/corpus_test_100.txt --annotation-timeout 600 --training-timeout 1200
```

**Résultat attendu :**
- Annotation complétée en < 5 minutes
- Fichiers créés dans /app/training_ner/data/ : train.json, val.json, test.json, label2id.json
- Logs indiquent "Annotation completed"

**Si échec :**
- Vérifier logs runtime dans console RunPod
- Vérifier que Jean-Baptiste/camembert-ner est accessible (connexion internet du pod)
- Vérifier CUDA/GPU disponible

---

### Étape 3: Test Training (10-20 minutes avec mini dataset)
Si l'annotation du mini-corpus a réussi, le training devrait démarrer automatiquement (avec workflow.py).

**Résultat attendu :**
- Training complété en < 15 minutes (mini dataset)
- Modèle sauvegardé dans /app/artifacts/
- Logs indiquent "Training completed"

**Si échec :**
- Vérifier que le config /app/training_ner/configs/kd_camembert.yaml existe
- Vérifier les chemins data dans le config (doivent pointer vers /app/training_ner/data/)
- Vérifier VRAM suffisante (batch_size dans config)

---

### Étape 4: Workflow Complet Production (1-3 heures)
Une fois le test réussi avec mini-corpus, lancer le workflow complet :

```powershell
python workflow.py --corpus-path /workspace/corpus_fr_100k_medico_FINAL.txt --annotation-timeout 7200 --training-timeout 14400
```

**Résultat attendu :**
- Annotation : 30-60 minutes (96k phrases)
- Training : 1-2 heures (dépend du GPU et hyperparamètres)
- Artefacts dans /app/artifacts/ : modèle distillé + métriques

---

## 🆘 TROUBLESHOOTING

### Erreur : "FileNotFoundError: /workspace/corpus..."
- **Cause** : Chemin de montage du volume incorrect
- **Solution** : Essayer `/app/corpus/` ou `/runpod-volume/` au lieu de `/workspace/`
- **Vérification** : Dans console RunPod, ouvrir un pod temporaire et vérifier `ls /workspace` vs `ls /app` vs `ls /runpod-volume`

### Erreur : "ModuleNotFoundError: No module named 'training_ner'"
- **Cause** : Problème d'import ou __init__.py manquant
- **Solution** : Vérifier que __init__.py existe dans training_ner/ (déjà créé)
- **Workaround** : Ajouter `sys.path.insert(0, '/app')` au début du handler

### Erreur : "CUDA out of memory"
- **Cause** : Batch size trop élevé
- **Solution** : Modifier `kd_camembert.yaml` → réduire `batch_size` (ex: 16 → 8)

### Erreur : "Connection timeout" ou "Model download failed"
- **Cause** : Pas d'accès internet depuis le pod
- **Solution** : Vérifier settings réseau de l'endpoint, ou pré-télécharger le modèle dans l'image Docker

### Job reste en "IN_QUEUE" > 2 minutes
- **Cause** : Aucun worker disponible (cold start)
- **Solution** : Attendre le démarrage du worker (30-60s). Si > 5 min, vérifier settings de l'endpoint (workers min/max)

---

## 📊 MÉTRIQUES DE SUCCÈS

### Upload Corpus
- ✅ Job complété < 2 minutes
- ✅ Fichier vérifié sur volume (taille et nombre de lignes corrects)

### Annotation (mini-corpus 100 lignes)
- ✅ Complété < 5 minutes
- ✅ 3 fichiers JSON créés (train/val/test)
- ✅ label2id.json contient les entités (PER, LOC, ORG, etc.)

### Training (mini-dataset)
- ✅ Complété < 20 minutes
- ✅ Modèle sauvegardé (fichiers .bin/.safetensors dans /app/artifacts)
- ✅ Pas d'erreur CUDA/OOM

### Workflow Complet (96k phrases)
- ✅ Annotation complétée < 1 heure
- ✅ Training complété < 3 heures
- ✅ Modèle distillé créé et téléchargeable
- ✅ Métriques F1-score affichées dans logs

---

## ⚡ COMMANDES RAPIDES DE RÉFÉRENCE

```powershell
# Config (une fois par session PowerShell)
$env:RUNPOD_ENDPOINT_ID = "wupg1xsork5mk7"
$env:RUNPOD_API_KEY = "VOTRE_CLE_API_ICI"

# Upload corpus
python upload_corpus.py corpus\corpus_fr_100k_medico_FINAL.txt /workspace/corpus_fr_100k_medico_FINAL.txt

# Monitor un job
python monitor_jobs.py "job-id-ici" --timeout 3600

# Workflow complet
python workflow.py --corpus-path /workspace/corpus_fr_100k_medico_FINAL.txt

# Workflow training seul (si annotation déjà faite)
python workflow.py --corpus-path /workspace/corpus.txt --skip-annotation

# Test rapide endpoint
python test_endpoint.py
```

---

## 🎯 APRÈS LE SUCCÈS

1. **Télécharger les artefacts** depuis /app/artifacts/ (modèle distillé)
2. **Évaluer le modèle** sur un jeu de test manuel
3. **Comparer performances** teacher vs student (F1-score, taille, vitesse)
4. **Arrêter/supprimer l'endpoint** si plus besoin (économiser coûts)
5. **Documenter les hyperparamètres** qui ont fonctionné

---

**PRÊT POUR LE PUSH !** ✅
