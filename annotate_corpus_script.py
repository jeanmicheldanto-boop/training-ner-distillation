import json
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import os

# --- Configuration ---
MODEL_ID = "jmdanto/titibongbong_camembert-ner-fp16"
CORPUS_PATH = "corpus/corpus_local_120k.txt"
OUTPUT_PATH = "corpus/annotated_corpus.jsonl"
BATCH_SIZE = 16 # Réduit pour plus de sécurité sur la mémoire

# --- Fonctions ---

def sentence_generator(file_path):
    """
    Générateur qui lit un fichier texte ligne par ligne,
    en ignorant les lignes vides.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                yield stripped_line

def main():
    print("--- Step 1: Initializing model and tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ner_pipeline = pipeline(
        "token-classification",
        model=MODEL_ID,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0
    )
    print("Model and tokenizer loaded.")

    print(f"--- Step 2: Annotating corpus from {CORPUS_PATH} (streaming) ---")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Utilise le générateur pour traiter les phrases sans tout charger en mémoire
    sentence_iterator = sentence_generator(CORPUS_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        # tqdm ne peut pas connaître la longueur totale, donc pas de barre de progression
        print("Processing sentences... (no progress bar due to streaming)")
        
        # Traite les phrases par lots
        for result_batch in ner_pipeline(sentence_iterator, batch_size=BATCH_SIZE):
            for result_list in result_batch:
                # Si le pipeline retourne une liste de listes
                if isinstance(result_list, list):
                    for single_result in result_list:
                        f_out.write(json.dumps(single_result) + "\n")
                # Si le pipeline retourne une liste de dicts
                else:
                    f_out.write(json.dumps(result_list) + "\n")

    print(f"--- Step 3: Annotation complete. Output saved to {OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()