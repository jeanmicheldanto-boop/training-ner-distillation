"""
RunPod Handler for NER Distillation Training
Handles requests to train the NER distillation model
"""
import os
import json
import torch
from pathlib import Path
from training_ner.train_kd import main as train_main
from training_ner.annotate_corpus import main as annotate_main

import runpod
from training_ner.train_kd import main as train_main
from training_ner.annotate_corpus import main as annotate_main

def handler(job):
    """
    Handle RunPod requests for NER distillation training
    
    job["input"] should contain:
    {
        "action": "train" or "annotate",
        "corpus_path": "/path/to/corpus.txt" (for annotate),
        "config": "/path/to/config.yaml" (for train),
        "output_dir": "/path/to/output"
    }
    """
    try:
        job_input = job.get("input", {})
        action = job_input.get("action", "train")
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if gpu_available else "GPU: Not available (CPU mode)"
        
        if action == "annotate":
            # Auto-annotation with teacher model
            corpus_path = job_input.get("corpus_path", "/app/corpus/corpus_test.txt")
            output_dir = job_input.get("output_dir", "/app/training_ner/data")
            
            print(f"Starting annotation for {corpus_path}...")
            # Build arguments for the annotation script
            args = [
                '--corpus_path', corpus_path,
                '--output_dir', output_dir,
                '--model_name', 'Jean-Baptiste/camembert-ner',
                '--batch_size', '32'
            ]
            result = annotate_main(args)
            print("Annotation finished.")
            
            return {
                "status": "success",
                "message": f"Annotation completed. {gpu_info}",
                "result": result
            }
        
        elif action == "train":
            # Training
            config_path = job_input.get("config", "/app/training_ner/configs/kd_camembert.yaml")
            
            print(f"Starting training with config {config_path}...")
            # Build arguments for the training script
            args = [
                '--config', config_path
            ]
            train_main(args)
            print("Training finished.")
            
            return {
                "status": "success",
                "message": f"Training completed. {gpu_info}"
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action}. Use 'train' or 'annotate'"
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }

# Initialize the RunPod serverless worker
runpod.serverless.start({"handler": handler})


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
