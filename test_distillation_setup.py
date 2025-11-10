"""
Script de test pour vérifier la configuration de la distillation
"""
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
import json

print("=" * 80)
print("TEST 1: Vérification du corpus annoté")
print("=" * 80)

# Charger quelques exemples du corpus
with open("corpus/annotated_corpus.jsonl") as f:
    examples = [json.loads(line) for i, line in enumerate(f) if i < 5]

print(f"\nNombre d'exemples chargés: {len(examples)}")
for i, ex in enumerate(examples[:3]):
    if ex['entities']:
        print(f"\nExemple {i}:")
        print(f"  Text: {ex['text'][:80]}...")
        print(f"  Entities: {ex['entities']}")

print("\n" + "=" * 80)
print("TEST 2: Conversion en tokens + ner_tags")
print("=" * 80)

def convert_entities_to_tokens(example):
    text = example["text"]
    tokens = text.split()
    ner_tags = ["O"] * len(tokens)
    
    for entity in example.get("entities", []):
        entity_word = entity["word"]
        entity_label = entity["entity_group"]
        
        entity_start = text.find(entity_word)
        if entity_start == -1:
            continue
        
        entity_end = entity_start + len(entity_word)
        char_pos = 0
        for i, token in enumerate(tokens):
            token_start = char_pos
            token_end = char_pos + len(token)
            
            if not (token_end <= entity_start or token_start >= entity_end):
                ner_tags[i] = entity_label
            
            char_pos = text.find(token, char_pos) + len(token)
    
    return {"tokens": tokens, "ner_tags": ner_tags}

# Tester la conversion
for i, ex in enumerate(examples[:3]):
    if ex['entities']:
        converted = convert_entities_to_tokens(ex)
        print(f"\nExemple {i} converti:")
        print(f"  Tokens: {converted['tokens'][:10]}")
        print(f"  NER tags: {converted['ner_tags'][:10]}")
        tagged = [(t, tag) for t, tag in zip(converted['tokens'], converted['ner_tags']) if tag != 'O']
        print(f"  Tokens avec entités: {tagged}")

print("\n" + "=" * 80)
print("TEST 3: Labels découverts dans le dataset")
print("=" * 80)

raw_datasets = load_dataset("json", data_files="corpus/annotated_corpus.jsonl")
raw_datasets = raw_datasets.map(convert_entities_to_tokens)

unique_ner_tags = set()
for example in raw_datasets["train"]:
    unique_ner_tags.update(example["ner_tags"])

label_list = sorted(list(unique_ner_tags))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print(f"\nLabels découverts: {label_list}")
print(f"Nombre de labels: {len(label_list)}")
print(f"label2id: {label2id}")
print(f"id2label: {id2label}")

print("\n" + "=" * 80)
print("TEST 4: Chargement des modèles Teacher et Student")
print("=" * 80)

TEACHER_MODEL_ID = "jmdanto/titibongbong_camembert-ner-fp16"

teacher_model = AutoModelForTokenClassification.from_pretrained(
    TEACHER_MODEL_ID, num_labels=len(label_list), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

print(f"\nTeacher model loaded")
print(f"  Teacher config labels: {teacher_model.config.id2label}")
print(f"  Teacher num_labels: {teacher_model.config.num_labels}")

student_config = AutoConfig.from_pretrained(
    TEACHER_MODEL_ID, num_hidden_layers=11, num_labels=len(label_list), id2label=id2label, label2id=label2id
)
student_model = AutoModelForTokenClassification.from_config(student_config)

print(f"\nStudent model created")
print(f"  Student config labels: {student_model.config.id2label}")
print(f"  Student num_labels: {student_model.config.num_labels}")

# Copier les poids
student_model.roberta.embeddings.load_state_dict(teacher_model.roberta.embeddings.state_dict())
for i in range(student_config.num_hidden_layers):
    student_model.roberta.encoder.layer[i].load_state_dict(teacher_model.roberta.encoder.layer[i].state_dict())
student_model.classifier.load_state_dict(teacher_model.classifier.state_dict())

print("\nPoids copiés du teacher au student")

print("\n" + "=" * 80)
print("TEST 5: Test d'inférence sur un exemple")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained("camembert-base")

# Prendre un exemple avec entités
test_example = None
for ex in raw_datasets["train"]:
    if any(tag != "O" for tag in ex["ner_tags"]):
        test_example = ex
        break

if test_example:
    print(f"\nTexte de test: {test_example['text'][:100]}")
    print(f"Tokens: {test_example['tokens'][:10]}")
    print(f"NER tags attendus: {test_example['ner_tags'][:10]}")
    
    # Tokenize
    inputs = tokenizer(test_example["tokens"], is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
    
    # Prédiction du student
    student_model.eval()
    with torch.no_grad():
        outputs_student = student_model(**inputs)
        predictions_student = torch.argmax(outputs_student.logits, dim=2)[0]
    
    # Prédiction du teacher
    teacher_model.eval()
    with torch.no_grad():
        outputs_teacher = teacher_model(**inputs)
        predictions_teacher = torch.argmax(outputs_teacher.logits, dim=2)[0]
    
    print(f"\nPrédictions Student (premiers 10): {[id2label[p.item()] for p in predictions_student[:10]]}")
    print(f"Prédictions Teacher (premiers 10): {[id2label[p.item()] for p in predictions_teacher[:10]]}")
    
    # Comparer
    same = (predictions_student == predictions_teacher).sum().item()
    total = predictions_student.shape[0]
    print(f"\nAgreement Student/Teacher: {same}/{total} ({100*same/total:.1f}%)")

print("\n" + "=" * 80)
print("TEST 6: Vérification de la loss")
print("=" * 80)

# Créer un batch simple
test_batch = raw_datasets["train"].select(range(8))

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

tokenized_batch = test_batch.map(tokenize_and_align_labels, batched=True)

# Convertir en tensors
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_batch[i] for i in range(len(tokenized_batch))])

# Calculer la loss
student_model.train()
batch = {k: v.to(student_model.device) for k, v in batch.items()}
outputs = student_model(**batch)

print(f"\nLoss calculée: {outputs.loss.item()}")
print(f"Loss est NaN? {torch.isnan(outputs.loss).item()}")
print(f"Logits shape: {outputs.logits.shape}")
print(f"Logits min: {outputs.logits.min().item():.4f}")
print(f"Logits max: {outputs.logits.max().item():.4f}")
print(f"Logits mean: {outputs.logits.mean().item():.4f}")

print("\n" + "=" * 80)
print("RÉSUMÉ")
print("=" * 80)
print(f"✓ Corpus chargé et converti")
print(f"✓ Labels: {label_list}")
print(f"✓ Models chargés avec {len(label_list)} labels")
print(f"✓ Poids copiés du teacher au student")
if outputs.loss.item() > 0 and not torch.isnan(outputs.loss).item():
    print(f"✓ Loss calculée correctement: {outputs.loss.item():.4f}")
    print("\n🎉 Configuration OK - Prêt pour l'entraînement!")
else:
    print(f"✗ Problème avec la loss: {outputs.loss.item()}")
    print("\n⚠️ Il y a un problème à résoudre avant l'entraînement")
