# Pruned Student Model - Usage Guide

## Model Overview

**Base Model:** `camembert-ner-student-11L` (distilled from teacher 12L)
**Pruned Model:** `student_11L_pruned` (15% unstructured pruning)

### Performance Expectations
- **Before Pruning:** F1 = 85.45%, Accuracy = 99.03%
- **After Pruning (15%):** F1 ≈ 84-85%, Accuracy ≈ 98.5-99%
- **Sparsity:** 15% of attention + FFN weights set to zero

---

## 1. Loading the Model

### Standard Usage (No Changes Required)

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load pruned model - works exactly like unpruned version
model = AutoModelForTokenClassification.from_pretrained("./student_11L_pruned")
tokenizer = AutoTokenizer.from_pretrained("./student_11L_pruned")

# Inference
text = "Marie travaille à l'hôpital de Paris."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1)
```

**✓ No special code needed - works like any HuggingFace model**

---

## 2. Technical Specifications

### Pruning Details

| Aspect | Details |
|--------|---------|
| **Method** | Global magnitude-based unstructured pruning |
| **Sparsity** | 15% (85% of weights retained) |
| **Pruned Layers** | Attention (Q, K, V, output) + FFN (intermediate, output) |
| **Excluded Layers** | Embeddings, LayerNorm, Classifier head |
| **Format** | Dense tensors with zeros (not sparse format yet) |

### Model Architecture
- **Layers:** 11 encoder layers
- **Hidden size:** 768
- **Attention heads:** 12 per layer
- **Parameters:** ~110M (unchanged, but 15% are zeros)
- **Labels:** 5 NER tags (O, I-LOC, I-PER, I-MISC, I-ORG)

---

## 3. Storage & Size

### Current State
- **File size:** ~440 MB (same as unpruned - zeros still stored)
- **Actual size:** Could be ~375 MB with sparse format (~15% reduction)

### To Reduce Actual Size (Optional)

#### Option A: Convert to Sparse Format
```python
import torch
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("./student_11L_pruned")

# Convert weight matrices to sparse format
for name, module in model.named_modules():
    if hasattr(module, 'weight') and 'attention' in name or 'intermediate' in name:
        if (module.weight == 0).sum() > 0.1 * module.weight.numel():
            # Convert to sparse if >10% zeros
            sparse_weight = module.weight.to_sparse()
            # Note: Need custom inference code to use sparse weights
```

**⚠️ Warning:** Sparse format requires custom inference - not recommended unless critical.

#### Option B: Quantize to INT8
```python
# Further compress with quantization
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# Combined: 15% pruning + INT8 = ~60% size reduction, ~2-3% F1 loss
```

---

## 4. Inference Performance

### Speed Comparison
- **Unpruned Student:** 100% baseline
- **Pruned 15%:** ~102-105% speed (minimal change with dense format)
- **Pruned + Sparse:** ~110-120% speed (if sparse ops supported)
- **Pruned + INT8:** ~130-150% speed on CPU

### Memory Usage
- **Dense format:** Same as unpruned (~440 MB)
- **Sparse format:** ~375 MB (-15%)
- **INT8 quantized:** ~220 MB (-50%)

---

## 5. Fine-Tuning (If Needed)

If F1 drops below 84% after pruning, fine-tune for 0.5 epoch:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./student_11L_pruned_finetuned",
    num_train_epochs=0.5,
    per_device_train_batch_size=8,
    learning_rate=1e-5,  # Lower LR for fine-tuning
    weight_decay=0.01,
    fp16=False,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

**Expected:** Recover 50-80% of F1 loss within 0.5 epoch.

---

## 6. Production Deployment

### Recommended Configuration

```python
# Load model once at startup
model = AutoModelForTokenClassification.from_pretrained("./student_11L_pruned")
model.eval()  # Set to evaluation mode
model.to('cuda')  # Or 'cpu' depending on hardware

# For CPU deployment, consider INT8 quantization
if device == 'cpu':
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

# Batch inference for efficiency
def predict_batch(texts, batch_size=32):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    return outputs.logits.argmax(-1)
```

### Deployment Checklist
- ✓ Test F1 score on validation set (should be ≥84%)
- ✓ Measure inference latency on target hardware
- ✓ Check memory usage under load
- ✓ Consider INT8 quantization for CPU deployment
- ✓ Monitor predictions for quality degradation

---

## 7. Troubleshooting

### Model F1 Lower Than Expected
```bash
# Evaluate on full validation set
python train_distillation.py --model_path ./student_11L_pruned --eval_only

# If F1 < 84%: Fine-tune for 0.5 epoch
python train_distillation.py --model_path ./student_11L_pruned --num_epochs 0.5
```

### Loading Errors
```python
# If model fails to load
import torch
model = AutoModelForTokenClassification.from_pretrained(
    "./student_11L_pruned",
    ignore_mismatched_sizes=False,
    torch_dtype=torch.float32  # Ensure FP32
)
```

### Inference Too Slow
```python
# Enable optimizations
torch.set_num_threads(4)  # Adjust based on CPU cores
model.eval()
torch.set_grad_enabled(False)

# Consider ONNX export for production
# (separate guide needed)
```

---

## 8. Comparison Table

| Model | Layers | Sparsity | Size | F1 Score | Inference Speed |
|-------|--------|----------|------|----------|-----------------|
| Teacher (FP16) | 12 | 0% | ~220 MB | ~87% | Baseline |
| Student (FP32) | 11 | 0% | ~440 MB | 85.45% | 1.1x faster |
| **Student Pruned** | 11 | 15% | ~440 MB* | ~84-85% | 1.1x faster |
| Student Pruned + INT8 | 11 | 15% | ~220 MB | ~82-83% | 1.4x faster (CPU) |

*Dense format - can be reduced to ~375 MB with sparse format

---

## 9. Next Steps

### Immediate Actions
1. ✅ Download model from RunPod (`download_student.py`)
2. ✅ Apply 15% pruning (`prune_student.py --sparsity 0.15`)
3. ⏳ Evaluate pruned model (check F1 ≥ 84%)
4. ⏳ Upload to HuggingFace if satisfied

### Optional Improvements
- Fine-tune if F1 < 84%
- Try 20-25% sparsity if 15% maintains F1 > 84.5%
- Experiment with INT8 quantization for deployment
- Convert to ONNX for production inference

### Monitoring in Production
- Log prediction confidence scores
- Track F1 on live data samples
- Monitor inference latency
- Check for domain shift over time

---

## 10. Support & Resources

**Model Files:**
- `config.json` - Model architecture
- `model.safetensors` - Pruned weights
- `tokenizer.json` - CamemBERT tokenizer
- `pruning_info.json` - Pruning metadata

**Scripts:**
- `prune_student.py` - Apply pruning
- `train_distillation.py` - Evaluate/fine-tune
- `test_endpoint.py` - Test inference

**Key Hyperparameters:**
- Sparsity: 15% (conservative)
- Pruning method: Global magnitude (L1)
- Target layers: Attention + FFN only

---

**Last Updated:** November 10, 2025
**Model Version:** camembert-ner-student-11L-pruned-15pct
