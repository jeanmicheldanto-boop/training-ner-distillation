"""
Download the distilled student model from RunPod to local machine.
Run this on your LOCAL machine (not on RunPod).
"""

import os
from huggingface_hub import snapshot_download

# Configuration
RUNPOD_MODEL_PATH = "camembert-ner-student-11L"  # Path on RunPod
LOCAL_MODEL_PATH = "./student_11L_downloaded"     # Local save path

def download_from_runpod():
    """
    Instructions to download model from RunPod:
    
    Method 1 - Using RunPod web interface:
    1. Go to RunPod dashboard
    2. Open your pod's file browser
    3. Navigate to /workspace/training-ner-distillation/camembert-ner-student-11L/
    4. Select all files and download as ZIP
    5. Extract locally to ./student_11L_downloaded/
    
    Method 2 - Using SCP (if SSH enabled):
    Run on local machine:
    ```
    scp -r root@<runpod-ip>:/workspace/training-ner-distillation/camembert-ner-student-11L ./student_11L_downloaded
    ```
    
    Method 3 - Push to HuggingFace then download:
    Run on RunPod:
    ```
    python upload_to_hf.py --model_path camembert-ner-student-11L --repo_name <your-username>/camembert-ner-student-11L
    ```
    
    Then run on local machine:
    ```
    python download_student.py
    ```
    """
    print("=" * 60)
    print("STUDENT MODEL DOWNLOAD")
    print("=" * 60)
    print()
    print("Choose your download method:")
    print()
    print("1. RunPod Web Interface (easiest)")
    print("   - Open RunPod dashboard → File Browser")
    print("   - Download: camembert-ner-student-11L/ as ZIP")
    print("   - Extract to: ./student_11L_downloaded/")
    print()
    print("2. Direct download from HuggingFace (if uploaded)")
    repo_name = input("   Enter HF repo (e.g., username/model-name) or press Enter to skip: ").strip()
    
    if repo_name:
        print(f"\nDownloading from: {repo_name}")
        try:
            snapshot_download(
                repo_id=repo_name,
                local_dir=LOCAL_MODEL_PATH,
                local_dir_use_symlinks=False
            )
            print(f"✓ Model downloaded to: {LOCAL_MODEL_PATH}")
            verify_download()
        except Exception as e:
            print(f"✗ Download failed: {e}")
            print("Use Method 1 (Web Interface) instead.")
    else:
        print("\nUse RunPod web interface to download manually.")
        print(f"Extract files to: {os.path.abspath(LOCAL_MODEL_PATH)}")

def verify_download():
    """Verify all required model files are present."""
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model"
    ]
    
    print("\n" + "=" * 60)
    print("VERIFYING DOWNLOAD")
    print("=" * 60)
    
    missing = []
    for file in required_files:
        path = os.path.join(LOCAL_MODEL_PATH, file)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✓ {file} ({size:.2f} MB)")
        else:
            print(f"✗ {file} MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n⚠ Warning: {len(missing)} files missing!")
        print("Download may be incomplete.")
    else:
        print("\n✓ All files present! Model ready for pruning.")
        print(f"Model location: {os.path.abspath(LOCAL_MODEL_PATH)}")

if __name__ == "__main__":
    download_from_runpod()
