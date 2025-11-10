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
import evaluate
import os

# --- 1. Configuration ---
TEACHER_MODEL_ID = "jmdanto/titibongbong_camembert-ner-fp16"
STUDENT_MODEL_ID = "camembert-ner-student-11L"
ANNOTATED_CORPUS_PATH = "corpus/annotated_corpus_fixed.jsonl"
ALPHA_CE = 0.5
ALPHA_DISTILL = 0.5
TEMPERATURE = 2.0

# --- 2. Chargement des données ---
print("--- Step 2: Loading and preparing data ---")
raw_datasets = load_dataset("json", data_files=ANNOTATED_CORPUS_PATH)

# Le corpus est déjà au bon format (tokens + ner_tags), pas besoin de conversion
train_test_split = raw_datasets["train"].train_test_split(test_size=0.1, seed=42)
datasets = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})
print(f"Dataset loaded: {datasets}")

# --- 3. Découverte des labels ---
print("--- Step 3: Discovering labels and initializing models ---")

# Utiliser exactement le même mapping de labels que le teacher
teacher_config = AutoConfig.from_pretrained(TEACHER_MODEL_ID)
label2id = teacher_config.label2id
id2label = teacher_config.id2label
label_list = [id2label[i] for i in sorted(id2label.keys())]
num_labels = len(label_list)

print(f"Using teacher's label mapping: {label_list}")
print(f"Number of labels: {num_labels}")

# Utiliser le tokenizer de base camembert pour éviter les problèmes
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

# --- 4. Fonctions de préparation et métriques ---
def tokenize_and_align_labels(examples):
    # Les tokens sont déjà tokenizés par CamemBERT, on les convertit en input_ids
    input_ids = []
    attention_mask = []
    labels = []
    
    for tokens_list, ner_tags_list in zip(examples["tokens"], examples["ner_tags"]):
        # Convertir les tokens en IDs
        ids = tokenizer.convert_tokens_to_ids(tokens_list)
        
        # Tronquer si nécessaire
        if len(ids) > 512:
            ids = ids[:512]
            ner_tags_list = ner_tags_list[:512]
        
        input_ids.append(ids)
        attention_mask.append([1] * len(ids))
        
        # Convertir les tags NER en label IDs
        label_ids = []
        for tag in ner_tags_list:
            if tag in label2id:
                label_ids.append(label2id[tag])
            else:
                label_ids.append(-100)  # Ignorer les labels inconnus
        labels.append(label_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, remove_columns=datasets["train"].column_names)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- 5. Initialisation des modèles ---
print("--- Step 5: Initializing models ---")
teacher_model = AutoModelForTokenClassification.from_pretrained(
    TEACHER_MODEL_ID, num_labels=num_labels, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)
student_config = AutoConfig.from_pretrained(
    TEACHER_MODEL_ID, num_hidden_layers=11, num_labels=num_labels, id2label=id2label, label2id=label2id
)
student_model = AutoModelForTokenClassification.from_config(student_config)

student_model.roberta.embeddings.load_state_dict(teacher_model.roberta.embeddings.state_dict())
for i in range(student_config.num_hidden_layers):
    student_model.roberta.encoder.layer[i].load_state_dict(teacher_model.roberta.encoder.layer[i].state_dict())

# Copier aussi le classifier head du teacher
student_model.classifier.load_state_dict(teacher_model.classifier.state_dict())

print("Teacher and Student models initialized.")
print("\n--- DEBUG LABELS ---")
print("Model labels:", student_model.config.id2label)
print("Teacher labels:", teacher_model.config.id2label)
print("Dataset labels:", label_list)
print("label2id:", label2id)
print("id2label:", id2label)

# --- 6. Définition du Trainer de Distillation ---
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.to(self.args.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs_student = model(**inputs)
        loss_ce = outputs_student.loss
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_distill = loss_fct(
            F.log_softmax(outputs_student.logits / TEMPERATURE, dim=-1),
            F.softmax(outputs_teacher.logits / TEMPERATURE, dim=-1),
        ) * (TEMPERATURE ** 2)
        loss = ALPHA_CE * loss_ce + ALPHA_DISTILL * loss_distill
        return (loss, outputs_student) if return_outputs else loss

# --- 7. Entraînement ---
print("--- Step 7: Training ---")
training_args = TrainingArguments(
    output_dir=STUDENT_MODEL_ID,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,  # Désactivé pour éviter les problèmes de gradient
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    dataloader_num_workers=0,
    save_total_limit=2,  # Garder seulement les 2 meilleurs checkpoints
    report_to="none",  # Désactiver TensorBoard
)

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    teacher_model=teacher_model,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
trainer.save_model(STUDENT_MODEL_ID)
print(f"Training complete. Student model saved to '{STUDENT_MODEL_ID}'")