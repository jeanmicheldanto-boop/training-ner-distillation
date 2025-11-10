# -*- coding: utf-8 -*-
"""
Ce script annote un corpus texte en utilisant un modèle NER pré-entraîné (teacher).
Il génère un fichier JSONL qui servira de données d'entraînement pour la distillation.
"""

import torch
import json
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
import os

def main():
    # --- Configuration ---
    TEACHER_MODEL_ID = "jmdanto/titibongbong_camembert-ner-fp16"
    CORPUS_PATH = "corpus/corpus_local_120k.txt"
    OUTPUT_PATH = "corpus/annotated_corpus.jsonl"

    # S'assurer que le dossier de sortie existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Détection du device (GPU si disponible)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Utilisation du device: {'GPU' if device == 0 else 'CPU'}")

    # Chargement du corpus en mémoire
    print(f"Chargement du corpus depuis {CORPUS_PATH}...")
    try:
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        print(f"Corpus chargé : {len(sentences):,} phrases.")
    except FileNotFoundError:
        print(f"ERREUR: Le fichier corpus '{CORPUS_PATH}' n'a pas été trouvé. Assurez-vous qu'il est au bon endroit.")
        return

    # Chargement du pipeline NER et du tokenizer associé
    print(f"Chargement du pipeline NER pour le modèle : {TEACHER_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    ner_pipeline = pipeline(
        "ner",
        model=TEACHER_MODEL_ID,
        tokenizer=tokenizer,
        device=device,
        grouped_entities=True
    )
    print("Pipeline chargé avec succès.")

    # --- Annotation ---
    annotated_data = []
    print("\nDébut de l'annotation...")

    for sentence in tqdm(sentences, desc="Annotation du corpus"):
        if not sentence:
            continue
        
        try:
            # Exécution du pipeline
            ner_results = ner_pipeline(sentence)
            
            # Tokenisation de la phrase pour l'alignement
            # `encode` puis `convert_ids_to_tokens` est plus robuste
            inputs = tokenizer(sentence, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            tags = ['O'] * len(tokens)

            # Aligner les prédictions avec les tokens
            for entity in ner_results:
                start_char, end_char = entity['start'], entity['end']
                
                # La méthode char_to_token est la plus fiable pour l'alignement
                start_token_idx = inputs.char_to_token(start_char)
                end_token_idx = inputs.char_to_token(end_char - 1) # Le end_char est exclusif

                if start_token_idx is not None and end_token_idx is not None:
                    tags[start_token_idx] = f"B-{entity['entity_group']}"
                    for i in range(start_token_idx + 1, end_token_idx + 1):
                        tags[i] = f"I-{entity['entity_group']}"
            
            # Vérification de la cohérence
            if len(tokens) == len(tags) and len(tokens) > 0:
                annotated_data.append({
                    "tokens": tokens,
                    "ner_tags": tags
                })

        except Exception as e:
            print(f"\nErreur sur la phrase : '{sentence}' -> {e}")

    print(f"Annotation terminée. {len(annotated_data):,} phrases traitées.")

    # --- Sauvegarde ---
    if annotated_data:
        print(f"\nSauvegarde des données annotées dans {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for item in annotated_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("Sauvegarde terminée.")
    else:
        print("\nAucune donnée à sauvegarder.")

if __name__ == "__main__":
    main()
