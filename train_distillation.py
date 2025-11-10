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
ANNOTATED_CORPUS_PATH = "corpus/annotated_corpus.jsonl"
ALPHA_CE = 0.5
ALPHA_DISTILL = 0.5
TEMPERATURE = 2.0

# --- 2. Chargement des données ---
print("--- Step 2: Loading and preparing data ---")
raw_datasets = load_dataset("json", data_files=ANNOTATED_CORPUS_PATH)
train_test_split = raw_datasets["train"].train_test_split(test_size=0.1, seed=42)
datasets = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})
print(f"Dataset loaded: {datasets}")

# --- 3. Découverte des labels ---
print("--- Step 3: Discovering labels and initializing models ---")
unique_ner_tags = set(tag for example in datasets["train"] for tag in example["ner_tags"])
unique_ner_tags.update(set(tag for example in datasets["validation"] for tag in example["ner_tags"]))
label_list = sorted(list(unique_ner_tags))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
num_labels = len(label_list)
print(f"Discovered {num_labels} labels: {label_list}")

tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)

# --- 4. Fonctions de préparation et métriques ---
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
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
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
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
print("Teacher and Student models initialized.")

# --- 6. Définition du Trainer de Distillation ---
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.to(self.args.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
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
    # CORRECTION POUR LES ANCIENNES VERSIONS DE TRANSFORMERS
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
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