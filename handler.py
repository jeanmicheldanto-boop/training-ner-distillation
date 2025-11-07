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
            
            return {
                "status": "success",
                "message": f"Annotation started. {gpu_info}",
                "corpus": corpus_path,
                "output": output_dir,
                "gpu_available": gpu_available
            }
        
        elif action == "train":
            # Training
            config_path = job_input.get("config", "/app/training_ner/configs/kd_camembert.yaml")
            output_dir = job_input.get("output_dir", "/app/artifacts")
            
            os.makedirs(output_dir, exist_ok=True)
            
            return {
                "status": "success",
                "message": f"Training started. {gpu_info}",
                "config": config_path,
                "output": output_dir,
                "gpu_available": gpu_available
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
