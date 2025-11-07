# Training NER - Data Format

## Format des fichiers JSONL

Chaque ligne du fichier JSONL représente un exemple d'entraînement :

```json
{"tokens": ["Jean", "habite", "à", "Paris"], "ner_tags": ["B-PER", "O", "O", "B-LOC"]}
```

### Champs requis

- **tokens**: Liste de strings, tokens de la phrase
- **ner_tags**: Liste de strings, labels NER pour chaque token

### Schéma de labellisation

Le projet supporte le schéma **BIOES** (par défaut) ou **BIO**.

#### BIOES
- **B-TYPE**: Beginning (début d'entité)
- **I-TYPE**: Inside (continuation d'entité)
- **O**: Outside (pas une entité)
- **E-TYPE**: End (fin d'entité multi-tokens)
- **S-TYPE**: Single (entité d'un seul token)

#### BIO (alternatif)
- **B-TYPE**: Beginning
- **I-TYPE**: Inside
- **O**: Outside

### Types d'entités (exemple médico-social)

- **PER**: Personne (patients, médecins, travailleurs sociaux)
- **LOC**: Lieu (villes, adresses, établissements géographiques)
- **ORG**: Organisation (hôpitaux, EHPAD, associations, MDPH)
- **DATE**: Date (optionnel)
- **MISC**: Divers (optionnel)

## Fichiers de données

### train.jsonl
Données d'entraînement (80% du dataset)
- Minimum recommandé: 1000 exemples
- Optimal: 5000-10000 exemples

### val.jsonl
Données de validation (10% du dataset)
- Minimum: 100 exemples
- Optimal: 500-1000 exemples

### test.jsonl
Données de test (10% du dataset)
- Minimum: 100 exemples
- Optimal: 500-1000 exemples

### label2id.json
Mapping label → id numérique

```json
{
  "O": 0,
  "B-PER": 1,
  "I-PER": 2,
  "B-LOC": 3,
  "I-LOC": 4,
  ...
}
```

## Génération depuis votre corpus

Si vous avez un corpus non annoté (comme `corpus_fr_100k_medico_FINAL.txt`), vous devez :

1. **Annoter avec le teacher** :
```python
# Utiliser Jean-Baptiste/camembert-ner pour annoter automatiquement
# TODO: Script à créer (annotate_corpus.py)
```

2. **Vérifier qualité** : Valider échantillon manuellement

3. **Splitter** : train/val/test (80/10/10)

## Validation du format

Avant l'entraînement, vérifier le format :

```python
from data_loader import verify_data_format

verify_data_format("data/train.jsonl", num_examples=5)
```

## Exemples complets

### Exemple médico-social simple
```json
{"tokens": ["Le", "patient", "Jean", "Dupont", "consulte", "à", "l'", "hôpital", "."], "ner_tags": ["O", "O", "B-PER", "I-PER", "O", "O", "O", "B-ORG", "O"]}
```

### Exemple avec adresse
```json
{"tokens": ["EHPAD", "Les", "Roses", ",", "12", "rue", "de", "la", "Paix", ",", "Paris"], "ner_tags": ["B-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "I-LOC", "I-LOC", "I-LOC", "O", "B-LOC"]}
```

### Exemple avec professionnel
```json
{"tokens": ["Dr", "Martin", "travaille", "à", "la", "MDPH", "de", "Lyon"], "ner_tags": ["O", "B-PER", "O", "O", "O", "B-ORG", "O", "B-LOC"]}
```
