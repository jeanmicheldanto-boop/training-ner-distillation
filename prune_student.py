"""
Lightweight unstructured magnitude-based pruning for the distilled student model.
Prunes 15% of weights (very conservative) to maintain F1 > 84%.

Usage:
    python prune_student.py --model_path ./student_11L_downloaded --output_path ./student_11L_pruned --sparsity 0.15
"""

import torch
import argparse
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Configuration
SPARSITY = 0.15  # Conservative: prune only 15% of weights
DATA_PATH = "corpus/annotated_corpus_fixed.jsonl"
EVAL_BATCH_SIZE = 16

def parse_args():
    parser = argparse.ArgumentParser(description="Prune student NER model")
    parser.add_argument("--model_path", type=str, default="./student_11L_downloaded",
                        help="Path to downloaded student model")
    parser.add_argument("--output_path", type=str, default="./student_11L_pruned",
                        help="Path to save pruned model")
    parser.add_argument("--sparsity", type=float, default=0.15,
                        help="Sparsity ratio (0.15 = 15% weights pruned)")
    parser.add_argument("--evaluate", action="store_true", default=True,
                        help="Evaluate model before/after pruning (default: True)")
    parser.add_argument("--eval_samples", type=int, default=1000,
                        help="Number of samples for evaluation")
    return parser.parse_args()

def apply_global_magnitude_pruning(model, sparsity):
    """
    Apply global magnitude-based unstructured pruning.
    Prunes the smallest `sparsity` fraction of weights across all target layers.
    
    Targets:
    - Attention layers: query, key, value, output projections
    - FFN layers: intermediate and output dense layers
    
    Excludes:
    - Embeddings (critical for vocabulary)
    - LayerNorm (too few parameters, high impact)
    - Final classifier (task-critical)
    """
    print("\n" + "=" * 60)
    print(f"APPLYING GLOBAL MAGNITUDE PRUNING ({sparsity*100:.1f}% SPARSITY)")
    print("=" * 60)
    
    # Collect all parameters to prune
    parameters_to_prune = []
    total_params = 0
    
    for name, module in model.named_modules():
        # Target attention and FFN layers only
        if any(x in name for x in ['attention', 'intermediate', 'output']) and \
           not any(x in name for x in ['LayerNorm', 'classifier']):
            
            if hasattr(module, 'weight') and module.weight is not None:
                if len(module.weight.shape) >= 2:  # Only prune 2D+ tensors
                    parameters_to_prune.append((module, 'weight'))
                    total_params += module.weight.numel()
                    print(f"  → {name}.weight: {module.weight.shape}")
    
    print(f"\nTotal parameters to prune: {total_params:,}")
    print(f"Parameters that will be zeroed: {int(total_params * sparsity):,}")
    
    # Apply global magnitude pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    
    # Make pruning permanent (remove masks, apply zeros)
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    print("✓ Pruning applied and made permanent")
    
    return model

def calculate_sparsity(model):
    """Calculate actual sparsity of the model."""
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    sparsity = zero_params / total_params if total_params > 0 else 0
    return sparsity, zero_params, total_params

def evaluate_model(model, tokenizer, device, label2id, num_samples=1000):
    """Comprehensive evaluation with F1, precision, recall."""
    
    print("\n" + "=" * 60)
    print(f"EVALUATING MODEL (on {num_samples} samples)")
    print("=" * 60)
    
    # Load validation data
    dataset = load_dataset('json', data_files={'val': DATA_PATH}, split=f'val[:{num_samples}]')
    
    def tokenize_and_align(example):
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        
        # Convert CamemBERT tokens back to input_ids
        input_ids = []
        labels = []
        
        for token, tag in zip(tokens, ner_tags):
            if token in ['<s>', '</s>', '<pad>']:
                continue
            
            # Tokenize individual token
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            
            # First subtoken gets the label, rest get -100
            if token_ids:
                input_ids.extend(token_ids)
                label_id = label2id.get(tag, 0)
                labels.append(label_id)
                labels.extend([-100] * (len(token_ids) - 1))
        
        # Add special tokens
        input_ids = [tokenizer.bos_token_id] + input_ids[:126] + [tokenizer.eos_token_id]
        labels = [-100] + labels[:126] + [-100]
        
        # Pad
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < 128:
            input_ids.append(tokenizer.pad_token_id)
            attention_mask.append(0)
            labels.append(-100)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i % 200 == 0:
                print(f"  Progress: {i}/{num_samples}")
            
            batch = tokenize_and_align(example)
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu()[0]
            
            # Collect only non-padding positions
            for pred, label in zip(predictions, labels):
                if label != -100:
                    all_predictions.append(pred.item())
                    all_labels.append(label.item())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Per-class F1 for entities (exclude O tag)
    precision_ent, recall_ent, f1_ent, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', labels=[1,2,3,4], zero_division=0
    )
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'entity_precision': precision_ent,
        'entity_recall': recall_ent,
        'entity_f1': f1_ent,
        'num_samples': len(all_labels)
    }
    
    print("\nResults:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nEntity-only metrics (excluding O):")
    print(f"  Precision: {precision_ent:.4f}")
    print(f"  Recall:    {recall_ent:.4f}")
    print(f"  F1 Score:  {f1_ent:.4f}")
    
    return results

def save_pruned_model(model, tokenizer, output_path, sparsity_info):
    """Save pruned model with metadata."""
    print("\n" + "=" * 60)
    print("SAVING PRUNED MODEL")
    print("=" * 60)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save pruning metadata
    metadata = {
        "pruning_method": "global_magnitude_unstructured",
        "target_sparsity": sparsity_info['target'],
        "actual_sparsity": sparsity_info['actual'],
        "zero_params": sparsity_info['zeros'],
        "total_params": sparsity_info['total'],
        "pruned_layers": "attention + FFN (excluding embeddings, LayerNorm, classifier)",
        "usage_notes": {
            "loading": "Use AutoModelForTokenClassification.from_pretrained() as usual",
            "inference": "No special requirements - works like any HF model",
            "sparse_format": "Dense format with zeros - consider torch.sparse for production",
            "quantization_compatible": "Can be further quantized to INT8 if needed"
        }
    }
    
    with open(os.path.join(output_path, "pruning_info.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved to: {output_path}")
    print(f"✓ Sparsity: {sparsity_info['actual']*100:.2f}%")
    print(f"✓ Zero parameters: {sparsity_info['zeros']:,} / {sparsity_info['total']:,}")

def main():
    args = parse_args()
    
    print("=" * 60)
    print("STUDENT MODEL PRUNING")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Target sparsity: {args.sparsity*100:.1f}%")
    print()
    
    # Load model and tokenizer
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Get label mapping
    label2id = model.config.label2id
    print(f"✓ Model loaded: {model.config.num_hidden_layers} layers, {sum(p.numel() for p in model.parameters()):,} params")
    
    # Evaluate before pruning
    baseline_results = None
    if args.evaluate:
        print("\n" + "=" * 60)
        print("BASELINE MODEL PERFORMANCE")
        print("=" * 60)
        baseline_results = evaluate_model(model, tokenizer, device, label2id, args.eval_samples)
    
    # Calculate initial sparsity
    initial_sparsity, _, _ = calculate_sparsity(model)
    print(f"\nInitial sparsity: {initial_sparsity*100:.2f}%")
    
    # Apply pruning
    model = apply_global_magnitude_pruning(model, args.sparsity)
    
    # Calculate final sparsity
    final_sparsity, zero_params, total_params = calculate_sparsity(model)
    
    sparsity_info = {
        'target': args.sparsity,
        'actual': final_sparsity,
        'zeros': zero_params,
        'total': total_params
    }
    
    print(f"\n✓ Final sparsity: {final_sparsity*100:.2f}%")
    
    # Evaluate after pruning
    pruned_results = None
    if args.evaluate:
        print("\n" + "=" * 60)
        print("PRUNED MODEL PERFORMANCE")
        print("=" * 60)
        pruned_results = evaluate_model(model, tokenizer, device, label2id, args.eval_samples)
        
        # Compare results
        if baseline_results:
            print("\n" + "=" * 60)
            print("COMPARISON: BASELINE vs PRUNED")
            print("=" * 60)
            print(f"{'Metric':<20} {'Baseline':<12} {'Pruned':<12} {'Δ (abs)':<12} {'Δ (%)':<12}")
            print("-" * 68)
            
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'entity_f1']:
                baseline = baseline_results[metric]
                pruned = pruned_results[metric]
                delta_abs = pruned - baseline
                delta_pct = (delta_abs / baseline * 100) if baseline > 0 else 0
                
                metric_name = metric.replace('_', ' ').title()
                print(f"{metric_name:<20} {baseline:<12.4f} {pruned:<12.4f} {delta_abs:<+12.4f} {delta_pct:<+12.2f}%")
            
            # Recommendation
            print("\n" + "=" * 60)
            print("RECOMMENDATION")
            print("=" * 60)
            
            f1_loss = baseline_results['f1'] - pruned_results['f1']
            
            if f1_loss < 0.01:  # Less than 1% loss
                print("✓ EXCELLENT: F1 loss < 1% - pruning very successful!")
                print("  → Ready to deploy or try higher sparsity (20-25%)")
            elif f1_loss < 0.02:  # 1-2% loss
                print("✓ GOOD: F1 loss 1-2% - acceptable for most use cases")
                print("  → Consider fine-tuning 0.5 epoch to recover ~0.5-1% F1")
            elif f1_loss < 0.03:  # 2-3% loss
                print("⚠ MODERATE: F1 loss 2-3% - fine-tuning recommended")
                print("  → Fine-tune 0.5-1 epoch to recover performance")
            else:  # >3% loss
                print("✗ HIGH LOSS: F1 loss > 3% - pruning too aggressive")
                print("  → Reduce sparsity to 10% or fine-tune 1 epoch")
            
            sparsity_info['baseline_f1'] = baseline_results['f1']
            sparsity_info['pruned_f1'] = pruned_results['f1']
            sparsity_info['f1_loss'] = f1_loss
    
    # Save pruned model
    save_pruned_model(model, tokenizer, args.output_path, sparsity_info)
    
    print("\n" + "=" * 60)
    print("PRUNING COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Evaluate pruned model: python train_distillation.py --model_path ./student_11L_pruned --eval_only")
    print("2. If F1 > 84%: You're done! Upload to HuggingFace")
    print("3. If F1 < 84%: Fine-tune for 0.5 epoch to recover performance")
    print("\nUsage notes:")
    print("- Load like any HF model: AutoModelForTokenClassification.from_pretrained('./student_11L_pruned')")
    print("- No special inference code needed")
    print("- For production: Convert to torch.sparse format for real size reduction")
    print("- Compatible with further INT8 quantization if needed")

if __name__ == "__main__":
    main()
