"""
Script pour annoter automatiquement le corpus avec le teacher.
Utilise Jean-Baptiste/camembert-ner pour générer des données d'entraînement.

Usage:
    python annotate_corpus.py --input ../corpus/corpus_fr_100k_medico_FINAL.txt --output data/ --split 0.8 0.1 0.1

ATTENTION: Ce script génère des annotations SILVER (auto-générées).
Pour production, validation manuelle recommandée sur un échantillon.
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# TODO: Implémenter après installation dépendances
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from tqdm import tqdm
    DEPENDENCIES_OK = True
except ImportError as e:
    logging.warning(f"⚠️ Dependencies not installed: {e}")
    DEPENDENCIES_OK = False

from utils import setup_logging, set_seed

logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Annotate corpus with teacher NER model")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input corpus file (one sentence per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="Jean-Baptiste/camembert-ner",
        help="Teacher model to use for annotation",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help="Train/val/test split ratios (must sum to 1.0)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to annotate (None = all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for annotation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def load_corpus(filepath: str, max_samples: int = None) -> List[str]:
    """
    Charge le corpus (une phrase par ligne).
    
    Args:
        filepath: Chemin vers le fichier corpus
        max_samples: Nombre max de phrases à charger
        
    Returns:
        Liste de phrases
    """
    logger.info(f"📄 Loading corpus from {filepath}")
    
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
                
                if max_samples and len(sentences) >= max_samples:
                    break
    
    logger.info(f"  Loaded {len(sentences)} sentences")
    return sentences


def annotate_batch(
    sentences: List[str],
    model,
    tokenizer,
    id2label: Dict[int, str],
    device: str = "cuda",
) -> List[Dict]:
    """
    Annote un batch de phrases avec le teacher.
    
    TODO: Implémenter sur RunPod
    - Tokeniser sentences
    - Forward pass
    - Décoder predictions
    - Aligner tokens avec labels
    
    Args:
        sentences: Liste de phrases
        model: Modèle teacher
        tokenizer: Tokenizer
        id2label: Mapping id → label
        device: Device
        
    Returns:
        Liste de dicts {"tokens": [...], "ner_tags": [...]}
    """
    results = []
    
    # TODO: Implémenter annotation réelle
    # Pour l'instant, retourner format vide
    for sentence in sentences:
        # Tokeniser
        tokens = sentence.split()  # Tokenization simpliste
        ner_tags = ["O"] * len(tokens)  # Tous O par défaut
        
        results.append({
            "tokens": tokens,
            "ner_tags": ner_tags,
        })
    
    return results


def split_data(
    data: List[Dict],
    split_ratios: List[float],
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split données en train/val/test.
    
    Args:
        data: Liste de samples
        split_ratios: [train_ratio, val_ratio, test_ratio]
        seed: Random seed
        
    Returns:
        (train_data, val_data, test_data)
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    train_size = int(n * split_ratios[0])
    val_size = int(n * split_ratios[1])
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info(f"📊 Data split:")
    logger.info(f"  Train: {len(train_data)} ({len(train_data)/n:.1%})")
    logger.info(f"  Val: {len(val_data)} ({len(val_data)/n:.1%})")
    logger.info(f"  Test: {len(test_data)} ({len(test_data)/n:.1%})")
    
    return train_data, val_data, test_data


def save_jsonl(data: List[Dict], filepath: str):
    """
    Sauvegarde données en format JSONL.
    
    Args:
        data: Liste de dicts
        filepath: Chemin de sortie
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"💾 Saved {len(data)} samples to {filepath}")


def extract_labels(data: List[Dict]) -> Dict[str, int]:
    """
    Extrait tous les labels uniques et crée label2id mapping.
    
    Args:
        data: Liste de samples annotés
        
    Returns:
        label2id dict
    """
    labels = set()
    
    for sample in data:
        for tag in sample["ner_tags"]:
            labels.add(tag)
    
    # Trier avec O en premier
    sorted_labels = ["O"] + sorted([l for l in labels if l != "O"])
    
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    
    logger.info(f"📊 Found {len(label2id)} unique labels: {list(label2id.keys())}")
    
    return label2id


def main():
    """Fonction principale d'annotation."""
    args = parse_args()
    
    setup_logging()
    logger.info("=" * 80)
    logger.info("🏷️  Automatic Corpus Annotation")
    logger.info("=" * 80)
    
    if not DEPENDENCIES_OK:
        logger.error("❌ Dependencies not installed. Run: pip install -r requirements.txt")
        return
    
    set_seed(args.seed)
    
    # Charger corpus
    sentences = load_corpus(args.input, max_samples=args.max_samples)
    
    # TODO: Charger teacher et annoter
    logger.info("\n" + "=" * 80)
    logger.info("🤖 LOADING TEACHER MODEL")
    logger.info("=" * 80)
    logger.warning("⚠️ TODO: Implement teacher loading and annotation on RunPod")
    
    # Pour l'instant, générer données d'exemple
    logger.info("\n" + "=" * 80)
    logger.info("🏷️  ANNOTATING CORPUS")
    logger.info("=" * 80)
    
    annotated_data = []
    batch_size = args.batch_size
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # TODO: Remplacer par annotation réelle
        batch_annotated = annotate_batch(
            batch,
            model=None,  # TODO: passer vrai modèle
            tokenizer=None,  # TODO: passer vrai tokenizer
            id2label={},  # TODO: passer vrai mapping
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        annotated_data.extend(batch_annotated)
        
        if (i // batch_size + 1) % 100 == 0:
            logger.info(f"  Annotated {len(annotated_data)}/{len(sentences)} sentences")
    
    logger.info(f"✅ Annotation complete: {len(annotated_data)} sentences")
    
    # Split data
    logger.info("\n" + "=" * 80)
    logger.info("✂️  SPLITTING DATA")
    logger.info("=" * 80)
    
    train_data, val_data, test_data = split_data(
        annotated_data,
        split_ratios=args.split,
        seed=args.seed,
    )
    
    # Extract labels
    label2id = extract_labels(train_data)
    
    # Save files
    logger.info("\n" + "=" * 80)
    logger.info("💾 SAVING FILES")
    logger.info("=" * 80)
    
    output_dir = Path(args.output)
    
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")
    
    # Save label2id
    label2id_path = output_dir / "label2id.json"
    with open(label2id_path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Saved label mapping to {label2id_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ ANNOTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total samples: {len(annotated_data)}")
    logger.info(f"Train: {len(train_data)}")
    logger.info(f"Val: {len(val_data)}")
    logger.info(f"Test: {len(test_data)}")
    logger.info(f"Labels: {len(label2id)}")
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\n⚠️ IMPORTANT: These are SILVER annotations (auto-generated)")
    logger.info("Recommend manual validation on sample before training")


if __name__ == "__main__":
    main()
