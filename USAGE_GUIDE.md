# 🚀 Guide d'Utilisation - Workflow NER Distillation RunPod

Ce guide explique comment utiliser les scripts pour orchestrer et suivre votre workflow d'annotation et de training sur RunPod.

---

## 📁 Fichiers créés

1. **`monitor_jobs.py`** - Monitoring en temps réel d'un job spécifique
2. **`workflow.py`** - Orchestration complète du workflow (annotation → training)
3. **`test_endpoint.py`** - Tests manuels de l'endpoint (déjà existant, corrigé)
4. **`CORPUS_UPLOAD.md`** - Guide détaillé pour uploader le corpus

---

## ⚙️ Configuration initiale (à faire UNE FOIS par session PowerShell)

```powershell
# Définir les variables d'environnement
$env:RUNPOD_ENDPOINT_ID = "wupg1xsork5mk7"
$env:RUNPOD_API_KEY = "VOTRE_CLE_API_ICI"
```

**💡 Astuce:** Pour rendre ces variables persistantes entre sessions, ajoutez-les à votre profil PowerShell:
```powershell
notepad $PROFILE
# Ajoutez les deux lignes ci-dessus et sauvegardez
```

---

## 🎯 Scénarios d'utilisation

### 📋 Scénario 1: Workflow complet automatisé (RECOMMANDÉ)

**Utilisation:** Lancer annotation puis training automatiquement, avec suivi en temps réel.

```powershell
# 1. Uploadez d'abord le corpus (voir CORPUS_UPLOAD.md)
# 2. Lancez le workflow complet
python workflow.py --corpus-path /runpod-volume/corpus_fr_100k_medico_FINAL.txt
```

**Ce script va:**
- ✅ Soumettre le job d'annotation
- ⏳ Attendre la fin de l'annotation (avec progress updates toutes les minutes)
- ✅ Soumettre le job de training automatiquement
- ⏳ Attendre la fin du training
- 📊 Afficher les résultats et la durée totale

**Options avancées:**
```powershell
# Training seul (si annotation déjà faite)
python workflow.py --corpus-path /runpod-volume/corpus.txt --skip-annotation

# Avec timeouts personnalisés (en secondes)
python workflow.py --corpus-path /runpod-volume/corpus.txt --annotation-timeout 3600 --training-timeout 7200

# Aide complète
python workflow.py --help
```

---

### 🔍 Scénario 2: Monitoring d'un job spécifique

**Utilisation:** Suivre un job déjà lancé (depuis test_endpoint.py ou la console).

```powershell
# Suivre un job avec son ID
python monitor_jobs.py "fdb73da3-d662-42ec-98e1-bc4eaf5529e3-e2"

# Avec timeout personnalisé (2 heures)
python monitor_jobs.py "job-id-ici" --timeout 7200
```

**Ce script va:**
- 🔄 Interroger le statut du job toutes les 5 secondes
- 📊 Afficher les changements de statut (IN_QUEUE → IN_PROGRESS → COMPLETED)
- ⏱️ Afficher la durée écoulée et le temps d'exécution
- ✅ Afficher l'output final si le job réussit
- ❌ Afficher les erreurs si le job échoue

---

### 🧪 Scénario 3: Test manuel de l'endpoint

**Utilisation:** Tester rapidement l'endpoint sans orchestration.

```powershell
# Lance 2 jobs de test (annotation + training) et affiche les IDs
python test_endpoint.py
```

**Ce script va:**
- 📤 Soumettre un job d'annotation de test
- 📤 Soumettre un job de training de test
- 📊 Afficher les job IDs pour monitoring manuel
- ⏳ Vérifier le statut initial (après 3 secondes)

**Ensuite, suivez un job avec monitor_jobs.py:**
```powershell
python monitor_jobs.py "job-id-affiché-par-test"
```

---

## 🧪 Workflow de test recommandé (AVANT le gros corpus)

**Objectif:** Valider que tout fonctionne avant de lancer un job long et coûteux.

### Étape 1: Créer un mini-corpus de test

```powershell
# Créer un fichier de 100 lignes pour test rapide
Get-Content C:\Users\Lenovo\dataner\corpus\corpus_fr_100k_medico_FINAL.txt -TotalCount 100 | Out-File C:\Users\Lenovo\dataner\corpus\corpus_test_100.txt -Encoding UTF8
```

### Étape 2: Uploader le mini-corpus sur RunPod

Suivez les instructions dans `CORPUS_UPLOAD.md` pour uploader `corpus_test_100.txt` vers `/runpod-volume/corpus_test_100.txt`

### Étape 3: Lancer le test workflow

```powershell
python workflow.py --corpus-path /runpod-volume/corpus_test_100.txt
```

**Durée attendue:** 2-5 minutes (annotation rapide + training court)

**Si ça marche ✅:**
- Votre handler est fonctionnel
- Les imports et configs sont corrects
- Vous pouvez passer au gros corpus

**Si ça échoue ❌:**
- Lisez les logs d'erreur affichés
- Vérifiez les Runtime Logs dans la console RunPod
- Corrigez le problème avant de passer au gros corpus

### Étape 4: Lancer le workflow complet

```powershell
# Uploadez le gros corpus (voir CORPUS_UPLOAD.md)
# Puis lancez:
python workflow.py --corpus-path /runpod-volume/corpus_fr_100k_medico_FINAL.txt
```

**Durée attendue:** 1-3 heures (annotation ~30-60 min, training ~1-2h)

---

## 📊 Interpréter les statuts des jobs

### Statuts RunPod:

| Statut | Signification | Action |
|--------|---------------|--------|
| `IN_QUEUE` | Job en attente d'un worker disponible | Attendre (normal si pas de worker actif) |
| `IN_PROGRESS` | Job en cours d'exécution | Attendre, surveiller les logs |
| `COMPLETED` | Job terminé avec succès | ✅ Récupérer les outputs/artefacts |
| `FAILED` | Job échoué | ❌ Lire les logs d'erreur, corriger et relancer |
| `CANCELLED` | Job annulé manuellement | Vérifier pourquoi, relancer si nécessaire |

### Logs disponibles:

- **Console RunPod:** https://console.runpod.io/jobs?id=<job_id>
- **Output dans monitor_jobs.py:** Affiche le champ `output` du résultat JSON
- **Runtime logs:** Accessible dans la console RunPod (onglet Logs du job)

---

## 🆘 Résolution de problèmes

### ❌ "Not Found" (404)
**Cause:** Mauvais endpoint ID ou endpoint non Ready  
**Solution:** Vérifiez l'endpoint ID dans la console RunPod, attendez que l'état soit "Ready"

### ❌ "Unauthorized" (401/403)
**Cause:** API key invalide ou permissions insuffisantes  
**Solution:** Vérifiez que `$env:RUNPOD_API_KEY` est correcte et a les permissions "All" ou "Invoke"

### ❌ Job reste en "IN_QUEUE" indéfiniment
**Cause:** Aucun worker disponible (endpoint Serverless idle)  
**Solution:** Attendez qu'un worker démarre (cold start ~30-60s) ou augmentez le nombre de workers dans les settings de l'endpoint

### ❌ Job échoue avec "FileNotFoundError"
**Cause:** Corpus non uploadé ou chemin incorrect  
**Solution:** 
1. Vérifiez que le corpus est bien uploadé sur le volume
2. Utilisez un pod temporaire pour lister les fichiers:
   ```bash
   ls -la /runpod-volume/
   ```
3. Corrigez le chemin dans la commande `workflow.py`

### ❌ Job échoue avec "ImportError" ou "ModuleNotFoundError"
**Cause:** Dépendances manquantes dans l'image Docker  
**Solution:**
1. Vérifiez le `requirements.txt` dans le repo
2. Ajoutez les dépendances manquantes
3. Rebuild l'endpoint (push vers GitHub déclenche un rebuild automatique)

### ⏱️ Timeout atteint
**Cause:** Job prend plus de temps que le timeout configuré  
**Solution:**
- Augmentez le timeout:
  ```powershell
  python workflow.py --corpus-path /path --annotation-timeout 10800 --training-timeout 21600
  # (3h et 6h respectivement)
  ```
- Ou suivez le job dans la console RunPod (il continue même après timeout du script)

---

## 🎯 Prochaines étapes après succès

### 1. Télécharger les artefacts

Les modèles entraînés sont stockés dans `/app/artifacts` sur le volume RunPod.

**Via un pod temporaire:**
```bash
# Démarrez un pod avec le volume attaché
cd /app/artifacts
ls -lh
# Téléchargez via l'interface web ou SCP
```

**Via runpodctl:**
```powershell
.\runpodctl.exe receive data <VOLUME_ID>:/app/artifacts C:\Users\Lenovo\dataner\artifacts
```

### 2. Évaluer le modèle distillé

Comparez les performances teacher vs student sur un jeu de test:
- F1-score par entité (PER, LOC, ORG, MISC)
- Taille du modèle (MB)
- Vitesse d'inférence (tokens/sec)

### 3. Optimiser (si nécessaire)

Si les performances du student sont insuffisantes:
- Ajustez les hyperparamètres dans `kd_camembert.yaml`
- Augmentez le nombre d'époques
- Modifiez les pondérations des pertes (α, β, γ, δ)
- Relancez le training avec `--skip-annotation`

---

## 📚 Référence rapide des commandes

```powershell
# Configuration (une fois par session)
$env:RUNPOD_ENDPOINT_ID = "wupg1xsork5mk7"
$env:RUNPOD_API_KEY = "votre_clé"

# Workflow complet (annotation + training)
python workflow.py --corpus-path /runpod-volume/corpus_fr_100k_medico_FINAL.txt

# Training seul
python workflow.py --corpus-path /runpod-volume/corpus.txt --skip-annotation

# Monitoring d'un job
python monitor_jobs.py "job-id-ici"

# Test rapide de l'endpoint
python test_endpoint.py

# Créer un mini-corpus de test
Get-Content corpus\corpus_fr_100k_medico_FINAL.txt -TotalCount 100 | Out-File corpus\corpus_test_100.txt -Encoding UTF8
```

---

## 💡 Conseils d'optimisation

### Coûts:
- **Testez d'abord avec un petit corpus** pour valider le workflow
- **Utilisez GPU 3090** pour le training (bon ratio performance/prix)
- **Arrêtez les pods temporaires** dès que l'upload est terminé
- **Surveillez les workers actifs** dans l'endpoint Serverless (facturation à l'utilisation)

### Performance:
- **Compression:** Compressez le corpus avant upload si très volumineux
- **Batch size:** Ajustez dans `kd_camembert.yaml` selon la VRAM disponible
- **Workers:** Augmentez le nombre de workers si vous avez plusieurs jobs en parallèle

### Sécurité:
- **Ne commitez jamais** l'API key dans git
- **Utilisez des clés séparées** pour dev/prod
- **Révocation:** Révocez les clés après usage ou en cas de fuite

---

**Besoin d'aide?** Consultez `CORPUS_UPLOAD.md` pour l'upload du corpus, ou les logs dans la console RunPod pour débugger les erreurs runtime.
