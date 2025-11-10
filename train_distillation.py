
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
import numpy as np
import os

# --- 1. Configuration ---
# Modèles
TEACHER_MODEL_ID = "jmdanto/titibongbong_camembert-ner-fp16"
STUDENT_MODEL_ID = "camembert-ner-student-11L" # Nom du dossier de sortie pour le modèle étudiant

# Fichiers
ANNOTATED_CORPUS_PATH = "corpus/annotated_corpus.jsonl"

# Paramètres de distillation
ALPHA_CE = 0.5  # Poids pour la loss sur les vrais labels (Cross-Entropy)
ALPHA_DISTILL = 0.5  # Poids pour la loss de distillation (KL Divergence)
TEMPERATURE = 2.0  # Température pour adoucir les logits

# --- 2. Chargement et préparation des données ---
print("--- Step 2: Loading and preparing data ---")

# Charger le dataset depuis le fichier JSONL
raw_datasets = load_dataset("json", data_files=ANNOTATED_CORPUS_PATH)

# On a une seule split 'train', on la divise en train/eval
train_test_split = raw_datasets["train"].train_test_split(test_size=0.1, seed=42)
datasets = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

print(f"Dataset loaded: {datasets}")

# Charger le tokenizer du professeur
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)

# Récupérer les labels depuis la config du professeur
teacher_config = AutoConfig.from_pretrained(TEACHER_MODEL_ID)
label2id = teacher_config.label2id
id2label = teacher_config.id2label
num_labels = len(label2id)

def tokenize_and_align_labels(examples):
    """Fonction pour tokenizer le texte et aligner les labels."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_str = label[word_idx]
                label_ids.append(label2id[label_str])
            else:
                # C'est un sous-mot, on met -100 aussi
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# --- 3. Initialisation des modèles ---
print("--- Step 3: Initializing models ---")

# Charger le modèle professeur
teacher_model = AutoModelForTokenClassification.from_pretrained(
    TEACHER_MODEL_ID,
    id2label=id2label,
    label2id=label2id,
)

# Créer la configuration de l'étudiant en retirant une couche
student_config = AutoConfig.from_pretrained(
    TEACHER_MODEL_ID,
    num_hidden_layers=11,
    id2label=id2label,
    label2id=label2id,
)

# Initialiser l'étudiant avec la nouvelle configuration (poids aléatoires)
student_model = AutoModelForTokenClassification.from_config(student_config)

# Copier les poids du professeur vers l'étudiant pour les couches correspondantes
# C'est une étape cruciale pour une bonne initialisation
student_model.camembert.embeddings.load_state_dict(teacher_model.camembert.embeddings.state_dict())
for i in range(student_config.num_hidden_layers):
    student_model.camembert.encoder.layer[i].load_state_dict(teacher_model.camembert.encoder.layer[i].state_dict())

print("Teacher and Student models initialized.")
print(f"Teacher layers: {teacher_model.config.num_hidden_layers}")
print(f"Student layers: {student_model.config.num_hidden_layers}")


# --- 4. Définition du Trainer de Distillation ---
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.to(self.args.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # Calculer la loss de l'étudiant (cross-entropy standard)
        outputs_student = model(**inputs)
        loss_ce = outputs_student.loss

        # Obtenir les logits du professeur (en mode eval et sans calcul de gradient)
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # Calculer la loss de distillation (KL Divergence)
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_distill = loss_fct(
            F.log_softmax(outputs_student.logits / TEMPERATURE, dim=-1),
            F.softmax(outputs_teacher.logits / TEMPERATURE, dim=-1),
        ) * (TEMPERATURE ** 2)

        # Combinaison des deux losses
        loss = ALPHA_CE * loss_ce + ALPHA_DISTILL * loss_distill
        
        return (loss, outputs_student) if return_outputs else loss

# --- 5. Entraînement ---
print("--- Step 5: Training ---")

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir=STUDENT_MODEL_ID,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True, # Activer le mixed-precision training sur H100
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Initialiser le Trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    teacher_model=teacher_model,
)

# Lancer l'entraînement
print("Starting training...")
trainer.train()

# Sauvegarder le modèle final
trainer.save_model(STUDENT_MODEL_ID)
print(f"Training complete. Student model saved to '{STUDENT_MODEL_ID}'")
