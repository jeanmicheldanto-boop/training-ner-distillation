import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm
import json
import os

# --- Configuration ---
MODEL_ID = "jmdanto/titibongbong_camembert-ner-fp16"
CORPUS_PATH = "corpus/corpus_local_120k.txt"
OUTPUT_PATH = "corpus/annotated_corpus.jsonl"
BATCH_SIZE = 8  # Taille de lot encore plus petite pour une sécurité maximale

# --- Fonctions ---

def read_sentences_in_batches(file_path, batch_size):
    """Lit un fichier et retourne des lots (batchs) de phrases."""
    with open(file_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                batch.append(stripped_line)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch: # Renvoyer le dernier lot s'il n'est pas vide
            yield batch

def main():
    print("--- Step 1: Initializing model and tokenizer ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer from base CamemBERT to avoid corrupted tokenizer.json
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID).to(device)
    model.eval() # Mettre le modèle en mode évaluation
    print(f"Model loaded on device: {device}")

    print(f"--- Step 2: Annotating corpus from {CORPUS_PATH} in batches ---")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Compter le nombre total de lignes pour la barre de progression
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        num_lines = sum(1 for line in f if line.strip())

    # Créer le générateur de lots
    batch_generator = read_sentences_in_batches(CORPUS_PATH, BATCH_SIZE)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        with torch.no_grad(): # Désactiver le calcul de gradient pour économiser de la mémoire
            for batch in tqdm(batch_generator, total=-(num_lines // -BATCH_SIZE), desc="Annotating"):
                # 1. Tokenization
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                
                # 2. Prédiction du modèle
                logits = model(**inputs).logits
                
                # 3. Conversion des prédictions en labels
                predictions = torch.argmax(logits, dim=2)
                
                # 4. Traitement et écriture des résultats un par un
                for i in range(len(batch)):
                    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i])
                    word_ids = inputs.word_ids(batch_index=i)
                    
                    ner_tags = [model.config.id2label[p.item()] for p in predictions[i]]
                    
                    # Logique d'agrégation manuelle (similaire à "simple")
                    current_word = None
                    aggregated_entities = []
                    for token_idx, word_id in enumerate(word_ids):
                        if word_id is None or word_id == current_word:
                            continue
                        
                        current_word = word_id
                        label = ner_tags[token_idx]
                        
                        if label != "O":
                            # Simplification: on ne regroupe pas les B- et I- pour le moment,
                            # on écrit juste les entités détectées.
                            # C'est suffisant pour l'entraînement de la distillation.
                            aggregated_entities.append({
                                "entity_group": label,
                                "word": batch[i].split()[word_id] if word_id < len(batch[i].split()) else "[UNK]"
                            })
                    
                    # Écriture d'un objet JSON par phrase originale
                    output_record = {
                        "text": batch[i],
                        "entities": aggregated_entities
                    }
                    f_out.write(json.dumps(output_record) + "\n")

    print(f"--- Step 3: Annotation complete. Output saved to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()