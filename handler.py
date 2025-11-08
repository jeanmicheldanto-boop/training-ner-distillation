"""
RunPod Handler for NER Distillation Training
Handles requests to train the NER distillation model
"""
import os
import json
import sys
import torch
import base64
import gzip
import requests
from pathlib import Path
from training_ner.train_kd import main as train_main
from training_ner.annotate_corpus import main as annotate_main

import runpod

def handler(job):
    """
    Handle RunPod requests for NER distillation training
    
    job["input"] should contain:
    {
        "action": "train" | "annotate" | "upload_corpus" | "upload_corpus_gzip" | "download_from_url",
        
        # For train:
        "config": "/path/to/config.yaml",
        
        # For annotate:
        "corpus_path": "/path/to/corpus.txt",
        "output_dir": "/path/to/output",
        
        # For upload_corpus:
        "remote_path": "/path/to/save.txt",
        "content_b64": "base64_encoded_content",
        
        # For upload_corpus_gzip:
        "remote_path": "/path/to/save.txt",
        "content_b64": "base64_encoded_gzipped_content",
        
        # For download_from_url:
        "url": "https://example.com/corpus.txt",
        "remote_path": "/path/to/save.txt",
        "timeout": 600 (optional, default 600s)
    }
    """
    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "train")
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if gpu_available else "GPU: Not available (CPU mode)"
        
        if action == "upload_corpus":
            # Upload corpus file to volume
            remote_path = job_input.get("remote_path")
            content_b64 = job_input.get("content_b64")
            
            if not remote_path or not content_b64:
                return {
                    "status": "error",
                    "message": "upload_corpus requires 'remote_path' and 'content_b64'"
                }
            
            print(f"Uploading corpus to {remote_path}...")
            
            # Decode base64 content
            content = base64.b64decode(content_b64).decode('utf-8')
            lines = content.count('\n')
            size_kb = len(content.encode('utf-8')) / 1024
            
            # Create parent directory if needed
            Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(remote_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Verify
            if os.path.exists(remote_path):
                actual_size = os.path.getsize(remote_path) / 1024
                print(f"Upload successful: {remote_path} ({actual_size:.1f} KB, {lines} lines)")
                
                return {
                    "status": "success",
                    "message": f"Corpus uploaded to {remote_path}",
                    "lines": lines,
                    "size_kb": round(actual_size, 1),
                    "path": remote_path
                }
            else:
                return {
                    "status": "error",
                    "message": f"File not found after write: {remote_path}"
                }
        
        elif action == "upload_corpus_gzip":
            # Upload corpus file with gzip compression
            remote_path = job_input.get("remote_path")
            content_b64 = job_input.get("content_b64")
            
            if not remote_path or not content_b64:
                return {
                    "status": "error",
                    "message": "upload_corpus_gzip requires 'remote_path' and 'content_b64'"
                }
            
            print(f"Uploading compressed corpus to {remote_path}...")
            
            # Decode base64 then decompress gzip
            compressed_data = base64.b64decode(content_b64)
            content = gzip.decompress(compressed_data).decode('utf-8')
            
            lines = content.count('\n')
            size_kb = len(content.encode('utf-8')) / 1024
            compressed_kb = len(compressed_data) / 1024
            ratio = 100 * (1 - compressed_kb / size_kb) if size_kb > 0 else 0
            
            print(f"Decompressed: {compressed_kb:.1f} KB → {size_kb:.1f} KB (ratio: {ratio:.1f}%)")
            
            # Create parent directory if needed
            Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(remote_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Verify
            if os.path.exists(remote_path):
                actual_size = os.path.getsize(remote_path) / 1024
                print(f"Upload successful: {remote_path} ({actual_size:.1f} KB, {lines} lines)")
                
                return {
                    "status": "success",
                    "message": f"Compressed corpus uploaded and decompressed to {remote_path}",
                    "lines": lines,
                    "size_kb": round(actual_size, 1),
                    "compressed_kb": round(compressed_kb, 1),
                    "compression_ratio": round(ratio, 1),
                    "path": remote_path
                }
            else:
                return {
                    "status": "error",
                    "message": f"File not found after write: {remote_path}"
                }
        
        elif action == "download_from_url":
            # Download corpus from URL
            url = job_input.get("url")
            remote_path = job_input.get("remote_path")
            timeout = job_input.get("timeout", 600)
            
            if not url or not remote_path:
                return {
                    "status": "error",
                    "message": "download_from_url requires 'url' and 'remote_path'"
                }
            
            print(f"Downloading from {url} to {remote_path}...")
            
            # Create parent directory if needed
            Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download with streaming to handle large files
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            total_size = 0
            with open(remote_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            # Count lines (for text files)
            lines = 0
            try:
                with open(remote_path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
            except:
                lines = -1  # Binary file or encoding error
            
            size_kb = total_size / 1024
            print(f"Download successful: {remote_path} ({size_kb:.1f} KB, {lines} lines)")
            
            return {
                "status": "success",
                "message": f"File downloaded from {url} to {remote_path}",
                "lines": lines if lines > 0 else "N/A (binary file)",
                "size_kb": round(size_kb, 1),
                "path": remote_path
            }
        
        elif action == "annotate":
            # Auto-annotation with teacher model
            corpus_path = job_input.get("corpus_path", "/app/corpus/corpus_test.txt")
            output_dir = job_input.get("output_dir", "/app/training_ner/data")
            
            print(f"Starting annotation for {corpus_path}...")
            
            # Set sys.argv for argparse in annotate_main
            sys.argv = [
                'annotate_corpus.py',
                '--input', corpus_path,
                '--output', output_dir,
                '--model_name', 'Jean-Baptiste/camembert-ner',
                '--batch_size', '32'
            ]
            
            result = annotate_main()
            print("Annotation finished.")
            
            return {
                "status": "success",
                "message": f"Annotation completed. {gpu_info}",
                "result": str(result)
            }
        
        elif action == "train":
            # Training
            config_path = job_input.get("config", "/app/training_ner/configs/kd_camembert.yaml")
            
            print(f"Starting training with config {config_path}...")
            
            # Set sys.argv for argparse in train_main
            sys.argv = [
                'train_kd.py',
                '--config', config_path
            ]
            
            train_main()
            print("Training finished.")
            
            return {
                "status": "success",
                "message": f"Training completed. {gpu_info}"
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action}. Use 'train', 'annotate', 'upload_corpus', 'upload_corpus_gzip', or 'download_from_url'"
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }


def main():
    """For local testing"""
    test_job = {
        "input": {
            "action": "train",
            "config": "/app/training_ner/configs/kd_camembert.yaml",
            "output_dir": "/app/artifacts"
        }
    }
    result = handler(test_job)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
else:
    # Initialize the RunPod serverless worker when imported as module
    runpod.serverless.start({"handler": handler})
