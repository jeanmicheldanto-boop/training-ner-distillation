#!/usr/bin/env python
"""
Quick validation script to test the entire pipeline without training.
Useful for pre-deployment checks.
"""

import sys
import json
import torch
from pathlib import Path

def check_environment():
    """Verify Python environment setup"""
    print("✓ Checking environment...")
    
    checks = {
        "Python 3.10+": sys.version_info >= (3, 10),
        "PyTorch 2.0+": torch.__version__.startswith("2."),
        "CUDA available": torch.cuda.is_available(),
        "CUDA version": torch.version.cuda is not None,
    }
    
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
    
    return all(checks.values())

def check_packages():
    """Verify required packages"""
    print("\n✓ Checking packages...")
    
    packages = [
        ("transformers", "4.30+"),
        ("torch_crf", "any"),
        ("numpy", "any"),
        ("tqdm", "any"),
        ("pyyaml", "any"),
    ]
    
    for pkg_name, version in packages:
        try:
            __import__(pkg_name.replace("_", "-"))
            print(f"  ✅ {pkg_name}")
        except ImportError:
            print(f"  ❌ {pkg_name} - NOT INSTALLED")
            return False
    
    return True

def check_models():
    """Verify teacher model accessibility"""
    print("\n✓ Checking model accessibility...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        # Just check if tokenizer can be loaded (doesn't download full model)
        tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner", local_files_only=False)
        print(f"  ✅ Teacher model (Jean-Baptiste/camembert-ner) accessible")
        return True
    except Exception as e:
        print(f"  ❌ Teacher model error: {str(e)[:100]}")
        return False

def check_data_format():
    """Verify data format expectations"""
    print("\n✓ Checking data format...")
    
    config_path = Path("configs/kd_camembert.yaml")
    label_path = Path("training_ner/data/label2id.json")
    
    if label_path.exists():
        with open(label_path) as f:
            labels = json.load(f)
            expected = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
            if labels == expected:
                print(f"  ✅ label2id.json format correct ({len(labels)} labels)")
                return True
            else:
                print(f"  ❌ label2id.json format mismatch")
                print(f"     Expected: {expected}")
                print(f"     Got: {labels}")
                return False
    else:
        print(f"  ⚠️  label2id.json not found (will be created)")
        return True

def check_config():
    """Verify configuration file"""
    print("\n✓ Checking configuration...")
    
    import yaml
    config_path = Path("configs/kd_camembert.yaml")
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            required_keys = ["teacher", "student", "loss_weights", "training", "pruning"]
            missing = [k for k in required_keys if k not in config]
            
            if missing:
                print(f"  ❌ Missing config keys: {missing}")
                return False
            
            print(f"  ✅ Config file valid with all required sections")
            return True
        except Exception as e:
            print(f"  ❌ Config parse error: {str(e)}")
            return False
    else:
        print(f"  ❌ Config file not found at {config_path}")
        return False

def check_gpu_memory():
    """Check GPU memory availability"""
    print("\n✓ Checking GPU memory...")
    
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available (CPU mode)")
        return True
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        available = total_memory - allocated
        
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total: {total_memory:.1f} GB")
        print(f"  Available: {available:.1f} GB")
        
        if available > 20:
            print(f"  ✅ Sufficient memory for training")
            return True
        else:
            print(f"  ⚠️  Low memory warning (need ~24GB for batch_size=16)")
            return False
    except Exception as e:
        print(f"  ⚠️  Could not read GPU memory: {str(e)}")
        return True

def check_disk_space():
    """Check available disk space"""
    print("\n✓ Checking disk space...")
    
    import shutil
    try:
        stat = shutil.disk_usage("/")
        total = stat.total / 1e9
        free = stat.free / 1e9
        used = stat.used / 1e9
        
        print(f"  Total: {total:.1f} GB")
        print(f"  Used: {used:.1f} GB")
        print(f"  Free: {free:.1f} GB")
        
        if free > 100:  # Need ~100GB for models, data, checkpoints
            print(f"  ✅ Sufficient disk space")
            return True
        else:
            print(f"  ⚠️  Low disk space warning (need ~100GB)")
            return False
    except Exception as e:
        print(f"  ⚠️  Could not check disk: {str(e)}")
        return True

def main():
    """Run all checks"""
    print("=" * 60)
    print("NER Distillation Pipeline - Pre-Deployment Validation")
    print("=" * 60)
    
    checks = [
        ("Environment", check_environment),
        ("Packages", check_packages),
        ("Models", check_models),
        ("Data Format", check_data_format),
        ("Configuration", check_config),
        ("GPU Memory", check_gpu_memory),
        ("Disk Space", check_disk_space),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_pass = all(r for _, r in results)
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ All checks passed! Ready for training.")
        print("\nNext steps:")
        print("1. Prepare corpus: cp corpus_fr_100k_medico_FINAL.txt data/")
        print("2. Auto-annotate: python training_ner/annotate_corpus.py ...")
        print("3. Train: python training_ner/train_kd.py ...")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nCommon issues:")
        print("- Install packages: pip install -r training_ner/requirements.txt")
        print("- Check GPU: nvidia-smi")
        print("- Download model: transformers-cli download Jean-Baptiste/camembert-ner")
        return 1

if __name__ == "__main__":
    exit(main())
