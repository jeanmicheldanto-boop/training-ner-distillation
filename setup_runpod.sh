#!/bin/bash
# Script de setup automatisé pour RunPod
# Usage: bash setup_runpod.sh

echo "🚀 RunPod Setup Script"
echo "======================="

# 1. Update système
echo "📦 Updating system..."
apt-get update -qq && apt-get upgrade -y -qq

# 2. Install dépendances
echo "📦 Installing Python dependencies..."
cd /workspace/ner-distillation

pip install --upgrade pip -q
pip install -r training_ner/requirements.txt -q

# 3. Vérifier CUDA
echo "🖥️  Checking GPU/CUDA..."
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 4. Valider setup
echo "✓ Validating setup..."
cd training_ner
python validate_setup.py

echo ""
echo "✅ Setup complete! Ready to start training."
echo ""
echo "Next steps:"
echo "1. Upload corpus: scp -P PORT your_corpus.txt root@IP:/workspace/ner-distillation/data/"
echo "2. Annotate: python annotate_corpus.py --input data/corpus.txt --output training_ner/data/"
echo "3. Train: python train_kd.py --config configs/kd_camembert.yaml --output artifacts/student_10L"
