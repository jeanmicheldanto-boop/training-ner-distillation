
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

# --- 3. Découverte des labels et initialisation des modèles ---
print("--- Step 3: Discovering labels and initializing models ---")

# Découvrir tous les labels uniques à partir du dataset
print("Discovering unique labels from the dataset...")
unique_ner_tags = set(tag for example in datasets["train"] for tag in example["ner_tags"])
unique_ner_tags.update(set(tag for example in datasets["validation"] for tag in example["ner_tags"]))
label_list = sorted(list(unique_ner_tags))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
num_labels = len(label_list)

print(f"Discovered {num_labels} labels: {label_list}")

# Charger le tokenizer du professeur
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)

# Charger le modèle professeur AVEC la nouvelle liste de labels
teacher_model = AutoModelForTokenClassification.from_pretrained(
    TEACHER_MODEL_ID,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True, # Important car la tête de classification peut ne pas correspondre
)

# Créer la configuration de l'étudiant en retirant une couche et AVEC les bons labels
student_config = AutoConfig.from_pretrained(
    TEACHER_MODEL_ID,
    num_hidden_layers=11,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# Initialiser l'étudiant avec la nouvelle configuration (poids aléatoires)
student_model = AutoModelForTokenClassification.from_config(student_config)

# Copier les poids du professeur vers l'étudiant pour les couches correspondantes
# C'est une étape cruciale pour une bonne initialisation
student_model.roberta.embeddings.load_state_dict(teacher_model.roberta.embeddings.state_dict())
for i in range(student_config.num_hidden_layers):
    student_model.roberta.encoder.layer[i].load_state_dict(teacher_model.roberta.encoder.layer[i].state_dict())

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
