#!/usr/bin/env python3
"""
Repair the saved student checkpoint by:
1. Loading the state dict (which has bert.* keys and no classifier)
2. Renaming bert.* -> roberta.* to match CamemBERT architecture
3. Adding classifier head from teacher (Jean-Baptiste/camembert-ner)
4. Saving as proper AutoModelForTokenClassification with correct config
"""

import torch
import os
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
)
from collections import OrderedDict

# Paths
BROKEN_CHECKPOINT = "/workspace/training-ner-distillation/artifacts/student_10L/pytorch_model.bin"
OUTPUT_DIR = "/workspace/training-ner-distillation/artifacts/student_10L_fixed"

# Label mapping
id2label = {0: "O", 1: "I-LOC", 2: "I-MISC", 3: "I-ORG", 4: "I-PER"}
label2id = {"O": 0, "I-LOC": 1, "I-MISC": 2, "I-ORG": 3, "I-PER": 4}

print("=" * 60)
print("STEP 1: Load broken checkpoint")
print("=" * 60)
state_dict = torch.load(BROKEN_CHECKPOINT, map_location="cpu")
print(f"✓ Loaded {len(state_dict)} keys")
print(f"  Sample keys: {list(state_dict.keys())[:3]}")
print(f"  Total params: {sum(v.numel() for v in state_dict.values()):,}")

print("\n" + "=" * 60)
print("STEP 2: Rename bert.* -> roberta.*")
print("=" * 60)
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith("bert."):
        new_key = key.replace("bert.", "roberta.", 1)
        new_state_dict[new_key] = value
        if len(new_state_dict) <= 3:
            print(f"  {key} -> {new_key}")
    else:
        new_state_dict[key] = value

print(f"✓ Renamed {len(new_state_dict)} keys")

print("\n" + "=" * 60)
print("STEP 3: Load teacher and extract classifier")
print("=" * 60)
teacher = AutoModelForTokenClassification.from_pretrained(
    "Jean-Baptiste/camembert-ner"
)
print(f"✓ Teacher loaded: {teacher.config.num_labels} labels")
print(f"  Classifier shape: {teacher.classifier.weight.shape}")

# Copy classifier weights (teacher has 5 labels, student needs 5)
new_state_dict["classifier.weight"] = teacher.classifier.weight.data.clone()
new_state_dict["classifier.bias"] = teacher.classifier.bias.data.clone()
print(f"✓ Added classifier layer: weight {new_state_dict['classifier.weight'].shape}, bias {new_state_dict['classifier.bias'].shape}")

print("\n" + "=" * 60)
print("STEP 4: Create config and instantiate model")
print("=" * 60)
# Use camembert-base config as template
config = AutoConfig.from_pretrained("camembert-base")
config.num_labels = 5
config.id2label = id2label
config.label2id = label2id
config.num_hidden_layers = 10  # your student has 10 layers (0-9)
config.architectures = ["CamembertForTokenClassification"]  # Fix architecture
print(f"✓ Config created: {config.num_hidden_layers} layers, {config.num_labels} labels")

# Instantiate empty model from config
model = AutoModelForTokenClassification.from_config(config)
print(f"✓ Model instantiated: {model.__class__.__name__}")

print("\n" + "=" * 60)
print("STEP 5: Load repaired state dict into model")
print("=" * 60)
# Load the repaired state dict
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print(f"✓ State dict loaded")
if missing:
    print(f"  ⚠ Missing keys ({len(missing)}): {missing[:5]}")
if unexpected:
    print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

print("\n" + "=" * 60)
print("STEP 6: Validate model structure")
print("=" * 60)
print(f"  Model layers: {len(model.roberta.encoder.layer)}")
print(f"  Classifier: {model.classifier.weight.shape}")
print(f"  Config labels: {model.config.num_labels}")

# Quick forward pass test
test_input = torch.randint(0, 1000, (1, 10))
with torch.no_grad():
    outputs = model(test_input)
print(f"✓ Forward pass successful")
print(f"  Input shape: {test_input.shape}")
print(f"  Logits shape: {outputs.logits.shape} (expected: [1, 10, 5])")
print(f"  Max prediction idx: {torch.argmax(outputs.logits).item()} (should be 0-4)")

print("\n" + "=" * 60)
print("STEP 7: Save repaired model")
print("=" * 60)
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
print(f"✓ Model saved to: {OUTPUT_DIR}")
print(f"  Files: {sorted(os.listdir(OUTPUT_DIR))}")

print("\n" + "=" * 60)
print("STEP 8: Test reload")
print("=" * 60)
# Verify the saved model can be loaded
reloaded = AutoModelForTokenClassification.from_pretrained(OUTPUT_DIR)
print(f"✓ Reloaded successfully: {reloaded.__class__.__name__}")
print(f"  Config: {reloaded.config.num_labels} labels, {reloaded.config.num_hidden_layers} layers")
print(f"  Classifier: {reloaded.classifier.weight.shape}")

# Test inference with reloaded model
with torch.no_grad():
    outputs2 = reloaded(test_input)
print(f"✓ Forward pass with reloaded model successful")
print(f"  Logits shape: {outputs2.logits.shape}")
print(f"  Max prediction: {torch.argmax(outputs2.logits).item()}")

print("\n" + "=" * 60)
print("✅ REPAIR COMPLETE!")
print("=" * 60)
print(f"Fixed model saved to: {OUTPUT_DIR}")
print("You can now use it with:")
print(f'  AutoModelForTokenClassification.from_pretrained("{OUTPUT_DIR}")')
print("Or with HF pipeline:")
print(f'  pipeline("ner", model="{OUTPUT_DIR}", aggregation_strategy="simple")')
