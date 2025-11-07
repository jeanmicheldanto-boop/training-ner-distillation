"""
Script de validation pré-RunPod.
Vérifie que tout est prêt avant déploiement.

Usage:
    python validate_setup.py
"""
import os
import sys
import json
import logging
from pathlib import Path

# Import local modules
try:
    from utils import setup_logging, load_config
    from data_loader import verify_data_format, load_label_mapping
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Make sure you're in the training_ner directory")
    sys.exit(1)

logger = logging.getLogger(__name__)


def check_file_exists(filepath: str, name: str) -> bool:
    """Vérifie qu'un fichier existe."""
    exists = os.path.exists(filepath)
    if exists:
        logger.info(f"✅ {name}: {filepath}")
    else:
        logger.error(f"❌ {name} NOT FOUND: {filepath}")
    return exists


def check_config(config_path: str) -> bool:
    """Vérifie la validité de la configuration."""
    logger.info("\n" + "=" * 60)
    logger.info("📋 CHECKING CONFIG")
    logger.info("=" * 60)
    
    try:
        config = load_config(config_path)
        logger.info(f"✅ Config loaded successfully")
        
        # Vérifier sections requises
        required_sections = ["teacher", "student", "distillation", "pruning", "training", "data"]
        for section in required_sections:
            if section in config:
                logger.info(f"  ✓ Section '{section}' found")
            else:
                logger.error(f"  ✗ Section '{section}' MISSING")
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Config error: {e}")
        return False


def check_data_files(config: dict) -> bool:
    """Vérifie que les fichiers de données existent et sont valides."""
    logger.info("\n" + "=" * 60)
    logger.info("📁 CHECKING DATA FILES")
    logger.info("=" * 60)
    
    data_config = config.get("data", {})
    
    all_ok = True
    
    # Vérifier fichiers JSONL
    for key in ["train_file", "val_file", "test_file"]:
        filepath = data_config.get(key)
        if filepath:
            exists = check_file_exists(filepath, key)
            all_ok = all_ok and exists
            
            if exists:
                # Compter lignes
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        num_lines = sum(1 for _ in f)
                    logger.info(f"    {num_lines} examples")
                except Exception as e:
                    logger.error(f"    Error reading file: {e}")
                    all_ok = False
        else:
            logger.error(f"❌ {key} not specified in config")
            all_ok = False
    
    # Vérifier label2id
    label2id_file = data_config.get("label2id_file")
    if label2id_file:
        exists = check_file_exists(label2id_file, "label2id_file")
        all_ok = all_ok and exists
        
        if exists:
            try:
                label2id, id2label = load_label_mapping(label2id_file)
                logger.info(f"    {len(label2id)} labels: {list(label2id.keys())}")
            except Exception as e:
                logger.error(f"    Error loading labels: {e}")
                all_ok = False
    else:
        logger.error(f"❌ label2id_file not specified in config")
        all_ok = False
    
    return all_ok


def check_data_format(config: dict) -> bool:
    """Vérifie le format des données."""
    logger.info("\n" + "=" * 60)
    logger.info("🔍 CHECKING DATA FORMAT")
    logger.info("=" * 60)
    
    data_config = config.get("data", {})
    train_file = data_config.get("train_file")
    
    if not train_file or not os.path.exists(train_file):
        logger.error("❌ Cannot verify format: train file missing")
        return False
    
    try:
        verify_data_format(train_file, num_examples=3)
        logger.info("✅ Data format looks good")
        return True
    except Exception as e:
        logger.error(f"❌ Data format error: {e}")
        return False


def check_dependencies() -> bool:
    """Vérifie que les dépendances sont installées."""
    logger.info("\n" + "=" * 60)
    logger.info("📦 CHECKING DEPENDENCIES")
    logger.info("=" * 60)
    
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "yaml",
        "numpy",
    ]
    
    all_ok = True
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} NOT INSTALLED")
            all_ok = False
    
    # Vérifier CUDA (optionnel)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning(f"⚠️ CUDA not available (will use CPU)")
    except:
        pass
    
    return all_ok


def check_teacher_accessible(config: dict) -> bool:
    """Vérifie que le modèle teacher est accessible."""
    logger.info("\n" + "=" * 60)
    logger.info("🔍 CHECKING TEACHER MODEL ACCESS")
    logger.info("=" * 60)
    
    teacher_config = config.get("teacher", {})
    model_name = teacher_config.get("model_name")
    
    if not model_name:
        logger.error("❌ teacher.model_name not specified in config")
        return False
    
    logger.info(f"Teacher model: {model_name}")
    
    try:
        # Tenter de charger juste la config (rapide)
        from transformers import AutoConfig
        config_hf = AutoConfig.from_pretrained(model_name)
        logger.info(f"✅ Teacher model accessible")
        logger.info(f"  Layers: {config_hf.num_hidden_layers}")
        logger.info(f"  Labels: {config_hf.num_labels}")
        return True
    except Exception as e:
        logger.error(f"❌ Cannot access teacher model: {e}")
        logger.warning(f"  Check internet connection or model name")
        return False


def main():
    """Fonction principale de validation."""
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("🔍 TRAINING SETUP VALIDATION")
    logger.info("=" * 60)
    
    # Vérifier config
    config_path = "configs/kd_camembert.yaml"
    config_ok = check_config(config_path)
    
    if not config_ok:
        logger.error("\n❌ Config validation FAILED")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Vérifier fichiers de données
    data_ok = check_data_files(config)
    
    # Vérifier format des données
    format_ok = check_data_format(config)
    
    # Vérifier dépendances
    deps_ok = check_dependencies()
    
    # Vérifier accès teacher
    teacher_ok = check_teacher_accessible(config)
    
    # Résumé
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    checks = {
        "Config": config_ok,
        "Data files": data_ok,
        "Data format": format_ok,
        "Dependencies": deps_ok,
        "Teacher access": teacher_ok,
    }
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        logger.info(f"{status} {check_name}")
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("✅ ALL CHECKS PASSED - Ready for RunPod!")
        logger.info("\nNext steps:")
        logger.info("1. Upload this folder to RunPod")
        logger.info("2. Install dependencies: pip install -r requirements.txt")
        logger.info("3. Run training: python train_kd.py")
        sys.exit(0)
    else:
        logger.error("❌ SOME CHECKS FAILED - Fix issues before RunPod")
        sys.exit(1)


if __name__ == "__main__":
    main()
