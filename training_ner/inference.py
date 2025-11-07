"""
Script d'inférence pour extraction NER.

Usage:
    python inference.py --model artifacts/student_10L_final --input phrases.txt --output entities.jsonl
"""
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict
import logging

from utils import setup_logging, get_device

logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Run NER inference on text")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (one sentence per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file with entities",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    
    return parser.parse_args()


def load_model(model_dir: str, device: str):
    """
    Charge le modèle et le tokenizer.
    
    TODO: Implémenter sur RunPod
    - Charger tokenizer
    - Charger modèle
    - Charger id2label mapping
    
    Args:
        model_dir: Dossier contenant le modèle
        device: Device pour inférence
        
    Returns:
        model, tokenizer, id2label
    """
    logger.info(f"📥 Loading model from {model_dir}")
    
    # TODO: Implémenter chargement réel
    tokenizer = None
    model = None
    id2label = {}
    
    logger.warning("⚠️ TODO: Implement model loading on RunPod")
    
    return model, tokenizer, id2label


def extract_entities(
    texts: List[str],
    model,
    tokenizer,
    id2label: Dict[int, str],
    device: str,
    batch_size: int = 16,
) -> List[Dict]:
    """
    Extrait les entités NER d'une liste de textes.
    
    TODO: Implémenter sur RunPod
    - Tokeniser textes
    - Forward pass par batches
    - Décoder prédictions
    - Extraire spans d'entités (BIOES → spans)
    
    Args:
        texts: Liste de phrases
        model: Modèle NER
        tokenizer: Tokenizer
        id2label: Mapping id → label
        device: Device
        batch_size: Taille des batches
        
    Returns:
        Liste de dicts avec texte et entités
    """
    results = []
    
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # TODO: Implémenter inférence réelle
            # 1. Tokeniser
            # 2. Forward pass
            # 3. Décoder labels
            # 4. Extraire spans
            
            for text in batch_texts:
                # Placeholder result
                result = {
                    "text": text,
                    "entities": [],  # TODO: Extraire entités réelles
                }
                results.append(result)
    
    return results


def bioes_to_spans(tokens: List[str], labels: List[str]) -> List[Dict]:
    """
    Convertit une séquence BIOES en spans d'entités.
    
    Args:
        tokens: Liste de tokens
        labels: Liste de labels BIOES
        
    Returns:
        Liste de dicts {"text": "...", "type": "PER", "start": 0, "end": 5}
    """
    entities = []
    current_entity = None
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            # Début d'une nouvelle entité
            if current_entity:
                entities.append(current_entity)
            
            entity_type = label[2:]
            current_entity = {
                "text": token,
                "type": entity_type,
                "start": i,
                "end": i + 1,
            }
        
        elif label.startswith("I-") and current_entity:
            # Continuation de l'entité
            current_entity["text"] += " " + token
            current_entity["end"] = i + 1
        
        elif label.startswith("E-") and current_entity:
            # Fin de l'entité
            current_entity["text"] += " " + token
            current_entity["end"] = i + 1
            entities.append(current_entity)
            current_entity = None
        
        elif label.startswith("S-"):
            # Entité single-token
            if current_entity:
                entities.append(current_entity)
            
            entity_type = label[2:]
            entities.append({
                "text": token,
                "type": entity_type,
                "start": i,
                "end": i + 1,
            })
            current_entity = None
        
        else:
            # O (Outside) ou autre
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Ajouter dernière entité si non terminée
    if current_entity:
        entities.append(current_entity)
    
    return entities


def main():
    """Fonction principale d'inférence."""
    args = parse_args()
    
    # Setup
    setup_logging()
    logger.info("=" * 80)
    logger.info("🔍 Starting NER Inference")
    logger.info("=" * 80)
    
    device = get_device()
    
    # Charger modèle
    logger.info("\n📥 Loading model...")
    model, tokenizer, id2label = load_model(args.model, device)
    
    # Charger textes
    logger.info(f"\n📄 Loading input from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    logger.info(f"  Loaded {len(texts)} texts")
    
    # Inférence
    logger.info("\n🔍 Running inference...")
    results = extract_entities(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
        batch_size=args.batch_size,
    )
    
    # Sauvegarder résultats
    logger.info(f"\n💾 Saving results to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ INFERENCE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Processed {len(texts)} texts")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
