# 📦 Guide d'Upload du Corpus vers RunPod

Ce document explique comment uploader le corpus `corpus_fr_100k_medico_FINAL.txt` (96 230 phrases) vers votre volume réseau RunPod pour pouvoir l'utiliser dans vos jobs d'annotation et de training.

---

## 🎯 Vue d'ensemble

**Fichier local:** `C:\Users\Lenovo\dataner\corpus\corpus_fr_100k_medico_FINAL.txt`  
**Destination RunPod:** `/runpod-volume/corpus_fr_100k_medico_FINAL.txt` (ou le chemin de votre volume réseau)

**Taille estimée:** ~10-20 MB (96k phrases)

---

## 🔧 Méthode 1: Via RunPod Web UI (Recommandée - Simple)

### Étapes:

1. **Connectez-vous à RunPod**
   - Allez sur https://www.runpod.io/console
   - Connectez-vous avec votre compte

2. **Accédez à votre Network Volume**
   - Dans le menu de gauche, cliquez sur **"Storage"** ou **"Network Volumes"**
   - Sélectionnez le volume attaché à votre endpoint (normalement créé automatiquement ou lors du setup de l'endpoint)
   - Notez le **Volume ID** et le **mount path** (généralement `/runpod-volume` ou `/workspace`)

3. **Démarrez un Pod temporaire (si pas déjà actif)**
   - Si aucun pod n'est actif avec accès au volume:
     - Allez dans **"Pods"** → **"+ Deploy"**
     - Choisissez une instance GPU simple (la moins chère suffit pour l'upload, ex: RTX 4000 Ada ou même CPU)
     - Dans **Storage**, attachez votre Network Volume
     - Déployez le pod (il démarrera en quelques secondes)

4. **Ouvrez le terminal du Pod**
   - Une fois le pod démarré, cliquez sur **"Connect"** → **"Start Web Terminal"** ou **"SSH Terminal"**
   - Vous aurez accès à un terminal Linux dans le pod

5. **Uploadez le fichier**
   
   **Option A: Via l'interface Web (drag-and-drop)**
   - Certains pods RunPod proposent un File Manager web (Jupyter, VS Code web, etc.)
   - Si disponible, ouvrez-le et drag-and-drop votre fichier `corpus_fr_100k_medico_FINAL.txt` vers `/runpod-volume/`

   **Option B: Via le terminal avec `curl` ou `wget` (si le fichier est en ligne)**
   - Si vous avez uploadé le corpus sur un service de stockage temporaire (WeTransfer, Dropbox, Google Drive avec lien public, etc.):
     ```bash
     cd /runpod-volume
     wget "URL_DU_FICHIER" -O corpus_fr_100k_medico_FINAL.txt
     ```

   **Option C: Via SCP/SFTP depuis votre machine locale**
   - Si le pod expose un port SSH, utilisez SCP depuis PowerShell:
     ```powershell
     scp C:\Users\Lenovo\dataner\corpus\corpus_fr_100k_medico_FINAL.txt root@<pod-ip>:/runpod-volume/
     ```
   - (Remplacez `<pod-ip>` par l'IP publique du pod visible dans la console RunPod)

6. **Vérifiez l'upload**
   - Dans le terminal du pod:
     ```bash
     ls -lh /runpod-volume/corpus_fr_100k_medico_FINAL.txt
     head -n 5 /runpod-volume/corpus_fr_100k_medico_FINAL.txt
     wc -l /runpod-volume/corpus_fr_100k_medico_FINAL.txt
     ```
   - Vous devriez voir le fichier avec ~96 230 lignes

7. **Arrêtez le pod temporaire (pour économiser)**
   - Une fois l'upload terminé, vous pouvez arrêter/supprimer ce pod temporaire
   - Le fichier restera sur le Network Volume et sera accessible par votre endpoint Serverless

---

## 🚀 Méthode 2: Via `runpodctl` CLI (Avancée - Plus rapide pour gros fichiers)

### Installation de `runpodctl`:

1. **Téléchargez `runpodctl` pour Windows**
   - Allez sur: https://github.com/runpod/runpodctl/releases
   - Téléchargez `runpodctl-windows-amd64.exe` (ou la version correspondante)
   - Renommez en `runpodctl.exe` et placez-le dans un dossier de votre PATH (ou dans `C:\Users\Lenovo\dataner`)

2. **Configurez `runpodctl`**
   ```powershell
   # Définir votre API key
   .\runpodctl.exe config --apiKey "votre_api_key_ici"
   
   # Lister vos volumes (pour trouver le Volume ID)
   .\runpodctl.exe get volume
   ```

3. **Uploadez le corpus**
   ```powershell
   # Syntaxe générale:
   # runpodctl send data <volume-id>:<destination-path> <local-path>
   
   .\runpodctl.exe send data <VOTRE_VOLUME_ID>:/corpus_fr_100k_medico_FINAL.txt C:\Users\Lenovo\dataner\corpus\corpus_fr_100k_medico_FINAL.txt
   ```

4. **Vérifiez l'upload**
   ```powershell
   .\runpodctl.exe exec <VOTRE_POD_ID> -- ls -lh /runpod-volume/corpus_fr_100k_medico_FINAL.txt
   ```

**Documentation officielle:** https://docs.runpod.io/cli/install-runpodctl

---

## 🧪 Méthode 3: Test avec un petit corpus d'abord (Recommandée avant le gros)

Pour tester le workflow sans attendre l'upload du gros corpus, créez un petit fichier de test:

### Sur votre machine locale:

```powershell
# Créer un mini-corpus de test (100 lignes)
Get-Content C:\Users\Lenovo\dataner\corpus\corpus_fr_100k_medico_FINAL.txt -TotalCount 100 | Out-File C:\Users\Lenovo\dataner\corpus\corpus_test_100.txt -Encoding UTF8
```

### Uploadez ce petit fichier de test:

- Utilisez l'une des méthodes ci-dessus pour uploader `corpus_test_100.txt` vers `/runpod-volume/corpus_test_100.txt`

### Lancez un test workflow:

```powershell
python workflow.py --corpus-path /runpod-volume/corpus_test_100.txt
```

- Cela vous permettra de vérifier que:
  - Le handler fonctionne correctement
  - L'annotation s'exécute sans erreur
  - Le training démarre après l'annotation
  - Les logs et outputs sont accessibles

**Une fois le test réussi, uploadez le gros corpus et relancez le workflow complet.**

---

## 📋 Checklist avant de lancer le workflow complet

- [ ] Corpus uploadé sur le volume RunPod
- [ ] Chemin vérifié (ex: `/runpod-volume/corpus_fr_100k_medico_FINAL.txt`)
- [ ] Variables d'environnement définies:
  ```powershell
  $env:RUNPOD_ENDPOINT_ID = "wupg1xsork5mk7"
  $env:RUNPOD_API_KEY = "votre_clé"
  ```
- [ ] Test avec petit corpus réussi (optionnel mais recommandé)
- [ ] Endpoint en état "Ready" dans la console RunPod
- [ ] Configuration `kd_camembert.yaml` présente dans l'image Docker (vérifiée lors du build)

---

## 🎯 Prochaines étapes après l'upload

### 1. Tester avec un petit corpus (recommandé):

```powershell
python workflow.py --corpus-path /runpod-volume/corpus_test_100.txt
```

### 2. Lancer le workflow complet:

```powershell
# Annotation + Training (workflow complet)
python workflow.py --corpus-path /runpod-volume/corpus_fr_100k_medico_FINAL.txt

# Ou si vous avez déjà annoté le corpus (training seul):
python workflow.py --corpus-path /runpod-volume/corpus_fr_100k_medico_FINAL.txt --skip-annotation
```

### 3. Suivre un job spécifique:

```powershell
# Si vous avez un job_id (ex: depuis test_endpoint.py ou workflow.py)
python monitor_jobs.py "fdb73da3-d662-42ec-98e1-bc4eaf5529e3-e2"

# Avec timeout personnalisé (2 heures = 7200 secondes)
python monitor_jobs.py "job-id-ici" --timeout 7200
```

---

## 🆘 Problèmes fréquents

### ❌ "FileNotFoundError: corpus not found"
- **Solution:** Vérifiez le chemin exact du corpus sur le volume. Utilisez un pod temporaire pour lister les fichiers:
  ```bash
  ls -la /runpod-volume/
  ```

### ❌ Upload très lent
- **Solution:** Utilisez `runpodctl` CLI (Méthode 2) qui est optimisé pour les transferts de fichiers volumineux
- Ou compressez le corpus avant upload:
  ```powershell
  Compress-Archive -Path C:\Users\Lenovo\dataner\corpus\corpus_fr_100k_medico_FINAL.txt -DestinationPath corpus.zip
  # Uploadez corpus.zip puis décompressez dans le pod:
  # unzip /runpod-volume/corpus.zip -d /runpod-volume/
  ```

### ❌ Volume non attaché à l'endpoint
- **Solution:** Vérifiez dans les settings de votre endpoint Serverless que le Network Volume est bien attaché. Si ce n'est pas le cas:
  - Allez dans Settings de l'endpoint
  - Sous "Storage", sélectionnez ou créez un Network Volume
  - Rebuild l'endpoint si nécessaire

---

## 💡 Astuces

- **Compression:** Si le corpus est très gros, compressez-le avant upload (gain de temps réseau)
- **Checksums:** Après upload, vérifiez l'intégrité:
  ```bash
  md5sum /runpod-volume/corpus_fr_100k_medico_FINAL.txt
  ```
  Comparez avec le checksum local (PowerShell):
  ```powershell
  Get-FileHash C:\Users\Lenovo\dataner\corpus\corpus_fr_100k_medico_FINAL.txt -Algorithm MD5
  ```

- **Encodage:** Assurez-vous que le fichier est en UTF-8 (important pour les accents français). Vérifiez dans le pod:
  ```bash
  file -i /runpod-volume/corpus_fr_100k_medico_FINAL.txt
  ```

---

## 📚 Ressources

- [Documentation RunPod Storage](https://docs.runpod.io/pods/storage/overview)
- [runpodctl CLI Docs](https://docs.runpod.io/cli/install-runpodctl)
- [Guide Serverless RunPod](https://docs.runpod.io/serverless/overview)

---

**Besoin d'aide?** Consultez les logs du pod ou de l'endpoint dans la console RunPod, ou exécutez `python monitor_jobs.py <job_id>` pour suivre en temps réel.
