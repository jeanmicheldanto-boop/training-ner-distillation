"""
DataLoader pour charger et prétraiter les données NER.
Format attendu: JSONL avec {"tokens": [...], "ner_tags": [...]}
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    """
    Dataset pour données NER en format JSONL.
    
    Format attendu:
    {"tokens": ["Jean", "habite", "à", "Paris"], "ner_tags": ["B-PER", "O", "O", "B-LOC"]}
    """
    
    def __init__(
        self,
        filepath: str,
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int = 512,
    ):
        """
        Args:
            filepath: Chemin vers fichier JSONL
            tokenizer: Tokenizer HuggingFace
            label2id: Mapping label -> id (ex: {"O": 0, "B-PER": 1, ...})
            max_length: Longueur max des séquences
        """
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
        # Charger données
        self.examples = self._load_examples()
        
        logger.info(f"📁 Loaded {len(self.examples)} examples from {filepath}")
    
    def _load_examples(self) -> List[Dict]:
        """
        Charge les exemples depuis le fichier JSONL.
        
        Returns:
            List de dicts {"tokens": [...], "ner_tags": [...]}
        """
        examples = []
        
        with open(self.filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    example = json.loads(line.strip())
                    
                    # Vérifier format
                    if "tokens" not in example or "ner_tags" not in example:
                        logger.warning(f"Line {i+1}: Missing 'tokens' or 'ner_tags', skipping")
                        continue
                    
                    if len(example["tokens"]) != len(example["ner_tags"]):
                        logger.warning(f"Line {i+1}: Mismatch tokens/tags length, skipping")
                        continue
                    
                    examples.append(example)
                
                except json.JSONDecodeError:
                    logger.warning(f"Line {i+1}: Invalid JSON, skipping")
                    continue
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retourne un exemple tokenisé avec labels alignés.
        
        Returns:
            Dict avec input_ids, attention_mask, labels
        """
        example = self.examples[idx]
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        
        # Tokeniser avec alignement des labels
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Aligner labels avec subword tokens
        labels = self._align_labels(
            tokens=tokens,
            ner_tags=ner_tags,
            word_ids=tokenized.word_ids(batch_index=0),
        )
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
    def _align_labels(
        self,
        tokens: List[str],
        ner_tags: List[str],
        word_ids: List[Optional[int]],
    ) -> List[int]:
        """
        Aligne les labels NER avec les subword tokens.
        
        Stratégie:
        - Premier subword d'un mot → label du mot
        - Subwords suivants → -100 (ignorés dans loss)
        - Tokens spéciaux (CLS, SEP, PAD) → -100
        
        Args:
            tokens: Tokens originaux
            ner_tags: Labels NER (strings)
            word_ids: Mapping subword → word index (None pour tokens spéciaux)
            
        Returns:
            List de label IDs alignés
        """
        labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            # Token spécial (CLS, SEP, PAD) → -100
            if word_idx is None:
                labels.append(-100)
            
            # Premier subword du mot → label du mot
            elif word_idx != previous_word_idx:
                label_str = ner_tags[word_idx]
                label_id = self.label2id.get(label_str, self.label2id["O"])
                labels.append(label_id)
            
            # Subwords suivants → -100
            else:
                labels.append(-100)
            
            previous_word_idx = word_idx
        
        return labels


def load_label_mapping(filepath: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Charge le mapping label2id et id2label depuis JSON.
    
    Args:
        filepath: Chemin vers label2id.json
        
    Returns:
        (label2id, id2label)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    
    id2label = {v: k for k, v in label2id.items()}
    
    logger.info(f"📊 Loaded {len(label2id)} labels: {list(label2id.keys())}")
    
    return label2id, id2label


def create_dataloaders(
    config: Dict,
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les dataloaders train/val/test.
    
    Args:
        config: Configuration YAML
        tokenizer: Tokenizer HuggingFace
        label2id: Mapping label -> id
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_config = config["data"]
    training_config = config["training"]
    
    # Créer datasets
    train_dataset = NERDataset(
        filepath=data_config["train_file"],
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=data_config["max_length"],
    )
    
    val_dataset = NERDataset(
        filepath=data_config["val_file"],
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=data_config["max_length"],
    )
    
    test_dataset = NERDataset(
        filepath=data_config["test_file"],
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=data_config["max_length"],
    )
    
    # Créer dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
        pin_memory=training_config["pin_memory"],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["dataloader_workers"],
        pin_memory=training_config["pin_memory"],
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["dataloader_workers"],
        pin_memory=training_config["pin_memory"],
    )
    
    logger.info(f"✅ Dataloaders created:")
    logger.info(f"  Train: {len(train_dataset)} examples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} examples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} examples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def verify_data_format(filepath: str, num_examples: int = 5):
    """
    Vérifie le format du fichier JSONL avant chargement.
    
    Args:
        filepath: Chemin vers fichier JSONL
        num_examples: Nombre d'exemples à afficher
    """
    logger.info(f"🔍 Verifying data format: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            
            try:
                example = json.loads(line.strip())
                logger.info(f"  Example {i+1}:")
                logger.info(f"    Tokens ({len(example.get('tokens', []))}): "
                            f"{example.get('tokens', [])[:10]}...")
                logger.info(f"    Tags ({len(example.get('ner_tags', []))}): "
                            f"{example.get('ner_tags', [])[:10]}...")
            
            except json.JSONDecodeError as e:
                logger.error(f"  Example {i+1}: JSON decode error: {e}")
    
    logger.info("✅ Data format verification complete")
