# NER Distillation Pipeline - French Medical/Social Domain

**Knowledge Distillation + Pruning for French CamemBERT-NER**

A production-ready knowledge distillation pipeline to compress the French CamemBERT-NER model (12 layers, 110M params) into a student model (10 layers, ~70M params) while maintaining performance on medical/social domain entity recognition.

## 🎯 Objectives

- **Reduce model size**: 110M → 70M parameters (-36%)
- **Maintain performance**: ≤ -1.0 F1 drop on end-to-end pipeline
- **Speed up inference**: 40-50% latency reduction expected
- **Stay practical**: Pragmatic tolerance for performance loss

## 📊 Dataset

- **96,230 French phrases** from diverse sources:
  - 46.5% medical/administrative documents (24,683 phrases)
  - 28.3% narrative texts (14,768 phrases)
  - 25.2% Wikipedia articles (13,380 phrases)
  - Plus 3,129 local injection examples
- **Total size**: 11.47 MB uncompressed
- **Format**: Plain text, auto-annotated via teacher model
- **Labels**: 7 classes (O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG)

## 🏗️ Architecture

### Teacher Model
- **Name**: Jean-Baptiste/camembert-ner
- **Layers**: 12
- **Parameters**: ~110M
- **Output**: Frozen during distillation (reference for soft labels + hidden states)
- **Role**: Provides golden soft labels and hidden state knowledge

### Student Model
- **Base**: CamemBERT (pretrained French RoBERTa)
- **Layers**: 10 (2 layers removed)
- **Parameters**: ~70M (after 25% attention head pruning)
- **Training**: Receives 4 combined losses from teacher
- **Inference**: Fast, deployable model

## 🔥 Distillation Strategy

**Patient Knowledge Distillation** with 4 loss components:

| Component | Purpose | Weight (Phase 1) | Weight (Phase 2+) |
|-----------|---------|------------------|-------------------|
| **L_CE** | Supervised classification loss | 1.0 | 1.0 |
| **L_KD** | KL divergence on logits (T=2.5) | 0.5 | 1.0 |
| **L_Hidden** | Cosine similarity on 6 hidden state pairs | 0.1 | 0.2 |
| **L_CRF** | L2 distance on CRF transition scores | 0.1 | 0.2 |

**Layer mapping for hidden distillation**:
- Teacher layer {2, 4, 6, 8, 10, 12} → Student layer {2, 3, 5, 7, 9, 10}

### Phase 1 (Epoch 1)
- Focus on task loss (L_CE) with weak distillation signals
- Weights: [1.0, 0.5, 0.1, 0.1]
- Allows student to stabilize on supervised signal

### Phase 2+ (Epochs 2-10)
- Full distillation engagement with strong hidden state matching
- Weights: [1.0, 1.0, 0.2, 0.2]
- Enables deep knowledge transfer

## 🔪 Pruning Strategy

**Structured Attention Head Pruning** (25% removal):

1. **Importance Metric**: $\text{Importance}_i = |\nabla_h \cdot a_i|$
   - Product of gradient (from loss) × activation (from forward pass)
   - Captures both task relevance and usage intensity

2. **Pruning**: Remove 2 of 8 heads per layer (25% reduction)
   - Cumulative speedup: ~15-20% across all layers
   - Combined with layer reduction: 40-50% end-to-end speedup

3. **Fine-tuning**: 2 additional epochs post-pruning
   - Adjusted weights: [1.0, 0.8, 0.15, 0.15]
   - Stabilizes remaining heads around new optimal configuration

## 📁 Project Structure

```
ner-distillation/
├── training_ner/
│   ├── models.py              # Teacher + Student architecture
│   ├── losses.py              # 4-way distillation loss
│   ├── pruning.py             # Attention head pruning logic
│   ├── data_loader.py         # Dataset + tokenization
│   ├── utils.py               # Logging, checkpointing, monitoring
│   ├── train_kd.py            # Main training script
│   ├── prune_heads.py          # Pruning orchestration
│   ├── finetune_postprune.py  # Post-pruning fine-tuning
│   ├── inference.py            # Inference + NER extraction
│   ├── annotate_corpus.py     # Auto-annotation via teacher
│   ├── validate_setup.py      # Pre-deployment validation
│   └── requirements.txt        # Python dependencies
├── configs/
│   └── kd_camembert.yaml      # Main configuration file
├── data/
│   ├── train.jsonl            # Training data (auto-generated)
│   ├── val.jsonl              # Validation data (auto-generated)
│   ├── test.jsonl             # Test data (auto-generated)
│   └── label2id.json          # Label encoding
├── artifacts/                 # Trained models (created at runtime)
├── setup_runpod.sh            # Automated RunPod setup
├── RUNPOD_GUIDE.md            # Detailed RunPod deployment guide
├── PROJECT_SUMMARY.md         # Technical strategy document
└── README.md                  # This file
```

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ with CUDA 11.8
- transformers, torch-crf, tqdm, pyyaml

### Installation

```bash
# Clone repository
git clone https://github.com/jeanmicheldanto/ner-distillation.git
cd ner-distillation

# Install dependencies
pip install -r training_ner/requirements.txt

# Validate setup
python training_ner/validate_setup.py
```

### Workflow

```bash
# 1. Prepare corpus (you must provide corpus_fr_100k_medico_FINAL.txt)
cp path/to/corpus_fr_100k_medico_FINAL.txt ./data/

# 2. Auto-annotate corpus using teacher model
python training_ner/annotate_corpus.py \
    --input data/corpus_fr_100k_medico_FINAL.txt \
    --output training_ner/data/ \
    --teacher_model Jean-Baptiste/camembert-ner \
    --batch_size 32

# 3. Train student via knowledge distillation
python training_ner/train_kd.py \
    --config configs/kd_camembert.yaml \
    --output artifacts/student_10L \
    --epochs 10 \
    --batch_size 16

# 4. Prune 25% attention heads
python training_ner/prune_heads.py \
    --model artifacts/student_10L/best_model \
    --pruning_ratio 0.25 \
    --output artifacts/student_10L_pruned

# 5. Fine-tune post-pruning
python training_ner/finetune_postprune.py \
    --model artifacts/student_10L_pruned/model \
    --config configs/kd_camembert.yaml \
    --output artifacts/student_10L_final \
    --epochs 2

# 6. Inference on test set
python training_ner/inference.py \
    --model artifacts/student_10L_final \
    --input training_ner/data/test.jsonl \
    --output results/predictions.jsonl
```

## 🐳 Running on RunPod (GPU)

See [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md) for detailed 5-phase deployment guide.

### Quick Setup

```bash
# Phase 0: Prepare on RunPod
bash setup_runpod.sh

# Phase 1: Create GitHub repo and push
# (Instructions in RUNPOD_GUIDE.md)

# Phase 2: SSH into RunPod pod
ssh root@<pod_ip>

# Phase 3: Clone + setup
cd /workspace
git clone https://github.com/YOUR_USERNAME/ner-distillation.git
cd ner-distillation

# Phase 4: Upload corpus
scp -P <PORT> corpus_fr_100k_medico_FINAL.txt root@<POD_IP>:/workspace/ner-distillation/data/

# Phase 5: Run pipeline
python training_ner/annotate_corpus.py --input data/corpus_fr_100k_medico_FINAL.txt --output training_ner/data/
python training_ner/train_kd.py --config configs/kd_camembert.yaml --output artifacts/student_10L
python training_ner/prune_heads.py --model artifacts/student_10L/best_model --output artifacts/student_10L_pruned
python training_ner/finetune_postprune.py --model artifacts/student_10L_pruned/model --output artifacts/student_10L_final

# Phase 6: Validate results
python training_ner/inference.py --model artifacts/student_10L_final --input training_ner/data/test.jsonl

# Download final model
# Copy artifacts/student_10L_final to your local machine
```

## 📊 Expected Results

| Metric | Teacher (12L, 110M) | Student (10L, 70M) | Student Pruned (10L-25%, 60M) | Δ F1 |
|--------|--------|--------|---------|------|
| **Inference Latency (ms)** | 150 | 130 | 90 | -40% |
| **Model Size (MB)** | 440 | 280 | 240 | -45% |
| **F1 Score** | 0.92 | ~0.91 | ~0.90 | ≤ -0.02 |
| **Precision** | 0.93 | ~0.92 | ~0.91 | - |
| **Recall** | 0.91 | ~0.90 | ~0.89 | - |

**Acceptance Criterion**: Final model's F1 drop on end-to-end 3-engine pipeline ≤ -1.0 (NER component drop can be -2 to -4)

## 🔧 Configuration

Edit `configs/kd_camembert.yaml` to modify:

```yaml
# Loss weights
loss_weights:
  ce: 1.0        # Cross-entropy on labels
  kd: 0.5        # KL divergence on logits
  hidden: 0.1    # Hidden state distillation
  crf: 0.1       # CRF transition loss

# Training hyperparameters
training:
  epochs: 10
  batch_size: 16
  learning_rate: 2e-5
  warmup_steps: 500
  gradient_clip: 1.0

# Pruning
pruning:
  method: "importance"  # gradient × activation
  ratio: 0.25          # Prune 25% of heads
```

## 💡 Troubleshooting

### OOM (Out of Memory)
- Reduce batch_size: `--batch_size 8`
- Reduce seq_length in config: `max_seq_length: 256`

### Slow training
- Verify GPU usage: `nvidia-smi`
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Poor results
- Verify teacher model is properly frozen (no gradients)
- Check loss weights are reasonable (not extreme ratios)
- Ensure data is properly BIOES-aligned (check `validate_setup.py`)

## 📚 References

- **CamemBERT**: Martin, L., et al. (2020) - French RoBERTa
- **Patient Knowledge Distillation**: Sun, S., et al. (2019) - IEEE ICCV
- **Structured Pruning**: See `pruning.py` for importance metric details

## 📝 Citation

```bibtex
@software{ner_distillation_2024,
  title={NER Distillation Pipeline for French Medical/Social Domain},
  author={Danto, Jean-Michel},
  year={2024},
  url={https://github.com/jeanmicheldanto/ner-distillation}
}
```

## ⚖️ License

MIT License - See LICENSE file

## 💬 Support

For issues, questions, or improvements:
- 📧 Email: jeanmichel.danto@gmail.com
- 🐙 GitHub: https://github.com/jeanmicheldanto

---

**Status**: ✅ Ready for production deployment on RunPod GPU
**Last Updated**: 2024
**Maintainer**: Jean-Michel Danto
