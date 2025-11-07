# Training Metrics & Monitoring Guide

## 📊 Expected Performance Metrics

### Phase 1: Knowledge Distillation (10 epochs)

**Epoch Progression**:
- **Epoch 1**: 
  - Train Loss: ~2.5-3.0 (high, learning signal establishment)
  - Val Loss: ~2.3-2.8
  - Val F1: ~0.75-0.80 (lower than teacher, but improving)

- **Epochs 2-5**:
  - Train Loss: ~1.8-2.2 (gradual decrease)
  - Val Loss: ~1.9-2.1
  - Val F1: ~0.82-0.86 (steady improvement)

- **Epochs 6-10**:
  - Train Loss: ~1.2-1.6 (stabilization)
  - Val Loss: ~1.4-1.7
  - Val F1: ~0.87-0.90 (approaching teacher performance)

**Target at End of Phase 1**:
- Student F1: **~0.90** (within -0.02 of teacher's 0.92)
- Student Loss: **~1.5** (stabilized)

### Phase 2: Head Pruning (25% removal)

**Pruning Statistics**:
- Heads to remove: 2 per layer × 12 layers = 24 heads removed
- Remaining heads: 72 heads (96 - 24)
- Expected inference speedup: **15-20%**
- Expected F1 drop: **-0.01 to -0.02**

**Post-Pruning F1**: ~0.88-0.89

### Phase 3: Fine-tuning Post-Pruning (2 epochs)

**Recovery Trajectory**:
- **Epoch 1 Post-Prune**:
  - Train Loss: ~1.8-2.0 (spike due to pruned heads)
  - Val F1: ~0.88-0.89 (recovering)

- **Epoch 2 Post-Prune**:
  - Train Loss: ~1.4-1.6 (stabilization)
  - Val F1: **~0.89-0.90** (recovered)

**Target at End of Phase 3**:
- Final Student F1: **~0.90**
- Pruned Student F1: **~0.90**
- Total inference speedup: **40-50%**
- Model size reduction: **45%** (440MB → 240MB)

## 🎯 Acceptance Criteria

✅ **GO** if on end-to-end 3-engine pipeline:
- F1 drop ≤ -1.0 (NER component alone can drop -2 to -4)
- Inference latency < 200ms per document
- Model runs without OOM on CPU

❌ **NO-GO** if:
- F1 drop > -1.0 on final pipeline
- Model cannot run on target hardware
- Inference latency > 300ms

## 📈 Monitoring Commands

### During Training

```bash
# Watch GPU usage
watch -n 2 nvidia-smi

# Monitor training in real-time
tail -f artifacts/student_10L/training.log | grep -E "Epoch|Loss|F1"

# Check checkpoint sizes
ls -lh artifacts/student_10L/checkpoints/

# Verify CUDA memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### After Training

```bash
# Compare metrics
python -c "
import json
with open('artifacts/student_10L/metrics.json') as f:
    metrics = json.load(f)
    print(f\"Best Epoch: {metrics['best_epoch']}\")
    print(f\"Best Val F1: {metrics['best_val_f1']:.4f}\")
    print(f\"Best Val Loss: {metrics['best_val_loss']:.4f}\")
"

# Model size comparison
ls -lh artifacts/student_10L*/model.bin | awk '{print $5, $9}'

# Inference speed test
python -c "
import torch
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained('artifacts/student_10L_final')
tokenizer = AutoTokenizer.from_pretrained('artifacts/student_10L_final')
model.eval()

text = 'Jean dupont habite à Paris depuis 2020.' * 50  # ~350 tokens
tokens = tokenizer(text, return_tensors='pt', truncation=True)

with torch.no_grad():
    start = time.time()
    for _ in range(100):
        outputs = model(**tokens)
    elapsed = (time.time() - start) / 100 * 1000
    print(f'Inference time: {elapsed:.2f}ms per document')
"
```

## 📊 CSV Logging Format

Training logs should follow this format for easy parsing:

```
epoch,train_loss,val_loss,val_f1,val_precision,val_recall,lr,pruning_ratio
1,2.845,2.412,0.756,0.768,0.745,2e-05,0.0
2,2.156,2.018,0.814,0.826,0.803,2e-05,0.0
3,1.954,1.876,0.852,0.861,0.843,2e-05,0.0
...
10,1.512,1.624,0.902,0.914,0.891,2e-05,0.0
11,1.847,1.955,0.889,0.895,0.883,2e-05,0.25  # Post-pruning
12,1.621,1.743,0.898,0.905,0.891,2e-05,0.25  # Fine-tuning done
```

## 🔍 Detailed Metric Analysis

### Per-Class Performance

Track these metrics per NER class (PER, LOC, ORG):

```
Class    | F1    | Precision | Recall | Support
---------|-------|-----------|--------|--------
O        | 0.98  | 0.98      | 0.98   | 45000
B-PER    | 0.88  | 0.89      | 0.87   | 2500
I-PER    | 0.86  | 0.87      | 0.85   | 2100
B-LOC    | 0.91  | 0.93      | 0.89   | 1800
I-LOC    | 0.89  | 0.91      | 0.87   | 1600
B-ORG    | 0.85  | 0.86      | 0.84   | 1200
I-ORG    | 0.83  | 0.84      | 0.82   | 900
```

### Loss Component Analysis

Monitor each loss component evolution:

```
Epoch | L_CE | L_KD | L_Hidden | L_CRF | Total
------|------|------|----------|-------|-------
1     | 1.2  | 0.8  | 0.45     | 0.35  | 2.80
2     | 0.9  | 0.6  | 0.32     | 0.28  | 2.10
...
10    | 0.6  | 0.4  | 0.18     | 0.12  | 1.30
```

## 🚨 Red Flags & Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Loss stuck at 3.0 | Teacher not frozen | Check `models.py` line 45: `teacher.requires_grad = False` |
| F1 = 0.0 | CRF transition error | Verify label encoding: `cat training_ner/data/label2id.json` |
| OOM after epoch 3 | GPU memory leak | Reduce batch_size, add `torch.cuda.empty_cache()` |
| Pruning hurts too much (-0.1+) | Pruning ratio too high | Reduce from 0.25 to 0.15 in config |
| Val loss increasing | Overfitting | Add dropout increase, reduce learning rate |

## 📅 Timeline Estimate

| Phase | Task | Duration | Resource |
|-------|------|----------|----------|
| 1 | Auto-annotation (96k phrases) | 2-4 hours | Teacher inference (RTX 4090) |
| 2 | Knowledge distillation (10 epochs) | 4-8 hours | Full GPU training (RTX 4090) |
| 3 | Head pruning + fine-tuning | 1.5-2.5 hours | Reduced training (RTX 4090) |
| 4 | Testing & metrics extraction | 30-45 min | CPU/GPU (small batch) |
| 5 | End-to-end pipeline validation | 2-4 hours | Production system |
| **Total** | **Complete workflow** | **~10-19 hours** | **RTX 4090 or A100** |

**Costs on RunPod**:
- RTX 4090: $0.70/hour × 15h avg = **~$10.50**
- A100: $1.50/hour × 15h avg = **~$22.50**
- With 50GB volume: +$0.10/hour = negligible

---

**Last Updated**: 2024-01-XX
**Owner**: Jean-Michel Danto
