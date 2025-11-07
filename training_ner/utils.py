"""
Fonctions utilitaires pour logging, checkpointing, monitoring.
"""
import os
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any
import random
import numpy as np


def setup_logging(level: str = "INFO"):
    """
    Configure le logging pour tout le projet.
    
    Args:
        level: Niveau de logging ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
    )


def load_config(config_path: str) -> Dict:
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier .yaml
        
    Returns:
        Dict avec configuration
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, output_path: str):
    """
    Sauvegarde la configuration dans un fichier YAML.
    
    Args:
        config: Configuration à sauvegarder
        output_path: Chemin de sortie
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def set_seed(seed: int):
    """
    Fixe la seed pour reproductibilité.
    
    Args:
        seed: Valeur de la seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Pour reproductibilité totale (peut ralentir)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Compte les paramètres du modèle.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Dict avec total, trainable, non-trainable
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def format_number(num: int) -> str:
    """
    Formate un nombre avec séparateurs (ex: 1234567 -> 1,234,567).
    
    Args:
        num: Nombre à formater
        
    Returns:
        String formatée
    """
    return f"{num:,}"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    output_dir: str,
    filename: str = "checkpoint.pt",
):
    """
    Sauvegarde un checkpoint d'entraînement.
    
    Args:
        model: Modèle à sauvegarder
        optimizer: Optimizer
        epoch: Epoch actuel
        step: Step actuel
        loss: Loss actuelle
        output_dir: Dossier de sortie
        filename: Nom du fichier
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
    
    filepath = os.path.join(output_dir, filename)
    torch.save(checkpoint, filepath)
    
    logging.info(f"💾 Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> Dict:
    """
    Charge un checkpoint d'entraînement.
    
    Args:
        filepath: Chemin vers le checkpoint
        model: Modèle où charger les poids
        optimizer: Optimizer où charger l'état (optionnel)
        
    Returns:
        Dict avec epoch, step, loss
    """
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    logging.info(f"📥 Checkpoint loaded: {filepath}")
    logging.info(f"  Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")
    
    return {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "loss": checkpoint["loss"],
    }


def save_model(
    model: torch.nn.Module,
    tokenizer,
    output_dir: str,
    config: Dict = None,
):
    """
    Sauvegarde le modèle final avec tokenizer et config.
    
    Args:
        model: Modèle à sauvegarder
        tokenizer: Tokenizer HuggingFace
        output_dir: Dossier de sortie
        config: Configuration du modèle (optionnel)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder modèle
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    logging.info(f"💾 Model saved: {model_path}")
    
    # Sauvegarder tokenizer
    tokenizer.save_pretrained(output_dir)
    logging.info(f"💾 Tokenizer saved: {output_dir}")
    
    # Sauvegarder config
    if config is not None:
        config_path = os.path.join(output_dir, "training_config.yaml")
        save_config(config, config_path)
        logging.info(f"💾 Config saved: {config_path}")
    
    # Sauvegarder config du modèle
    if hasattr(model, "config"):
        model.config.save_pretrained(output_dir)
        logging.info(f"💾 Model config saved: {output_dir}/config.json")


class TrainingMonitor:
    """
    Classe pour monitorer l'entraînement (loss, metrics, temps).
    """
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Dossier où sauvegarder les logs
        """
        self.output_dir = output_dir
        self.metrics = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.log_file = os.path.join(output_dir, "training_log.jsonl")
    
    def log(self, epoch: int, step: int, metrics: Dict[str, float]):
        """
        Log des métriques pour un step.
        
        Args:
            epoch: Epoch actuel
            step: Step actuel
            metrics: Dict avec métriques (loss, lr, etc.)
        """
        entry = {
            "epoch": epoch,
            "step": step,
            **metrics,
        }
        
        self.metrics.append(entry)
        
        # Écrire dans fichier JSONL
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    
    def get_summary(self) -> Dict:
        """
        Retourne un résumé des métriques.
        
        Returns:
            Dict avec moyennes et best values
        """
        if not self.metrics:
            return {}
        
        # Calculer moyennes
        keys = [k for k in self.metrics[0].keys() if k not in ["epoch", "step"]]
        summary = {}
        
        for key in keys:
            values = [m[key] for m in self.metrics if key in m]
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
        
        return summary
    
    def save_summary(self):
        """
        Sauvegarde le résumé dans un fichier JSON.
        """
        summary = self.get_summary()
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"📊 Training summary saved: {summary_path}")


def get_device() -> str:
    """
    Retourne le device disponible (cuda ou cpu).
    
    Returns:
        "cuda" si GPU disponible, sinon "cpu"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"🖥️  GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logging.info("⚠️  No GPU available, using CPU")
    
    return device


def format_time(seconds: float) -> str:
    """
    Formate un temps en secondes vers format lisible.
    
    Args:
        seconds: Temps en secondes
        
    Returns:
        String formatée (ex: "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
