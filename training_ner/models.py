"""
Définition des modèles Teacher et Student pour distillation NER.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TeacherModel(nn.Module):
    """
    Wrapper pour le modèle teacher (Jean-Baptiste/camembert-ner).
    Extrait logits, hidden states et transitions CRF pour distillation.
    """
    
    def __init__(self, model_name: str, check_crf: bool = True):
        """
        Args:
            model_name: Nom du modèle HuggingFace (ex: "Jean-Baptiste/camembert-ner")
            check_crf: Vérifier si le modèle contient une couche CRF
        """
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.config = self.model.config
        self.model.eval()  # Teacher toujours en mode eval
        
        # Vérifier présence CRF
        self.has_crf = check_crf and hasattr(self.model, 'crf')
        if self.has_crf:
            logger.info("✅ Teacher model has CRF layer")
            self.crf = self.model.crf
        else:
            logger.warning("⚠️ Teacher model does NOT have CRF layer")
            self.crf = None
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass du teacher (sans gradient).
        
        Returns:
            Dict contenant:
                - logits: (batch, seq_len, num_labels)
                - hidden_states: Tuple de (batch, seq_len, hidden_size)
                - crf_transitions: (num_labels, num_labels) si CRF disponible
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        result = {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
        }
        
        # Ajouter transitions CRF si disponibles
        if self.has_crf:
            result["crf_transitions"] = self.crf.transitions.data.clone()
        
        return result
    
    def get_crf_transitions(self) -> Optional[torch.Tensor]:
        """Retourne la matrice de transitions CRF du teacher."""
        if self.has_crf:
            return self.crf.transitions.data.clone()
        return None


class StudentModel(nn.Module):
    """
    Modèle student basé sur CamemBERT réduit à 10 couches.
    Copie les embeddings et la tête de classification du teacher.
    """
    
    def __init__(
        self,
        base_model: str,
        num_layers: int,
        num_labels: int,
        teacher_model: Optional[TeacherModel] = None,
        copy_embeddings: bool = True,
        copy_classifier: bool = True,
        copy_crf: bool = True,
    ):
        """
        Args:
            base_model: Modèle de base (ex: "camembert-base")
            num_layers: Nombre de couches (10 pour réduction)
            num_labels: Nombre de labels NER
            teacher_model: Modèle teacher pour copier poids
            copy_embeddings: Copier embeddings du teacher
            copy_classifier: Copier tête de classification du teacher
            copy_crf: Copier CRF du teacher si disponible
        """
        super().__init__()
        
        # Configuration student avec réduction de couches
        config = AutoConfig.from_pretrained(base_model)
        config.num_hidden_layers = num_layers
        config.num_labels = num_labels
        
        # Initialiser le modèle
        self.config = config
        self.bert = AutoModel.from_pretrained(base_model, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # TODO: Ajouter CRF layer si teacher en a un
        self.has_crf = False
        self.crf = None
        
        # Copier les poids du teacher si fourni
        if teacher_model is not None:
            self._copy_from_teacher(
                teacher_model,
                copy_embeddings=copy_embeddings,
                copy_classifier=copy_classifier,
                copy_crf=copy_crf,
            )
        
        logger.info(f"✅ Student model initialized with {num_layers} layers")
    
    def _copy_from_teacher(
        self,
        teacher: TeacherModel,
        copy_embeddings: bool,
        copy_classifier: bool,
        copy_crf: bool,
    ):
        """Copy weights from teacher to student with defensive checks.

        This implementation is tolerant to small API differences between
        transformer variants (missing token_type_embeddings, different
        LayerNorm attribute names, absent CRF, ...). It logs what was
        copied and warns on mismatches instead of throwing.
        """
        logger.info("🔄 Copying weights from teacher to student (defensive)")

        # Embeddings copy (if available)
        if copy_embeddings:
            teacher_emb = getattr(getattr(teacher.model, 'base_model', teacher.model), 'embeddings', None)
            student_emb = getattr(self.bert, 'embeddings', None)

            if teacher_emb is None or student_emb is None:
                logger.warning("  ⚠️ Embeddings object not found on teacher or student; skipping embeddings copy")
            else:
                try:
                    # word_embeddings
                    if hasattr(teacher_emb, 'word_embeddings') and hasattr(student_emb, 'word_embeddings'):
                        student_emb.word_embeddings.weight.data.copy_(teacher_emb.word_embeddings.weight.data)

                    # position_embeddings
                    if hasattr(teacher_emb, 'position_embeddings') and hasattr(student_emb, 'position_embeddings'):
                        student_emb.position_embeddings.weight.data.copy_(teacher_emb.position_embeddings.weight.data)

                    # token_type_embeddings (may be absent for RoBERTa/CamemBERT)
                    if hasattr(teacher_emb, 'token_type_embeddings') and hasattr(student_emb, 'token_type_embeddings'):
                        student_emb.token_type_embeddings.weight.data.copy_(teacher_emb.token_type_embeddings.weight.data)

                    # LayerNorm: try several common attribute names
                    ln_names = ['LayerNorm', 'layer_norm', 'layernorm', 'layernorm_embedding']
                    copied_ln = False
                    for name in ln_names:
                        if hasattr(teacher_emb, name) and hasattr(student_emb, name):
                            t_ln = getattr(teacher_emb, name)
                            s_ln = getattr(student_emb, name)
                            try:
                                s_ln.weight.data.copy_(t_ln.weight.data)
                                s_ln.bias.data.copy_(t_ln.bias.data)
                                copied_ln = True
                                break
                            except Exception:
                                # Continue searching other names
                                continue
                    if not copied_ln:
                        logger.warning("  ⚠️ Could not copy LayerNorm (no common attribute name found)")

                    logger.info("  ✅ Embeddings copied (best-effort)")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed during embeddings copy: {e}")

        # Classifier copy (weight + bias) - only if shapes compatible
        if copy_classifier:
            teacher_classifier = getattr(teacher.model, 'classifier', None)
            student_classifier = getattr(self, 'classifier', None)

            if teacher_classifier is None or student_classifier is None:
                logger.warning("  ⚠️ Classifier object missing on teacher or student; skipping classifier copy")
            else:
                try:
                    t_w_shape = tuple(teacher_classifier.weight.shape)
                    s_w_shape = tuple(student_classifier.weight.shape)
                    t_b_shape = tuple(teacher_classifier.bias.shape) if hasattr(teacher_classifier, 'bias') else None
                    s_b_shape = tuple(student_classifier.bias.shape) if hasattr(student_classifier, 'bias') else None

                    if t_w_shape == s_w_shape and t_b_shape == s_b_shape:
                        student_classifier.weight.data.copy_(teacher_classifier.weight.data)
                        if hasattr(teacher_classifier, 'bias') and hasattr(student_classifier, 'bias'):
                            student_classifier.bias.data.copy_(teacher_classifier.bias.data)
                        logger.info(f"  ✅ Classifier head copied (shape {s_w_shape})")
                    else:
                        logger.warning(f"  ⚠️ Classifier shape mismatch: teacher={t_w_shape}, student={s_w_shape}; skipping copy")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed during classifier copy: {e}")

        # CRF transitions copy: not supported in student implementation yet
        if copy_crf and getattr(teacher, 'has_crf', False):
            logger.info("  ⚠️ CRF copy not implemented for student; skipping CRF transitions")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass du student.
        
        Returns:
            Dict contenant:
                - logits: (batch, seq_len, num_labels)
                - loss: scalar (si labels fournis)
                - hidden_states: Tuple de tensors (si demandé)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        result = {
            "logits": logits,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
        }
        
        # Calculer loss si labels fournis
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten pour calcul de loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            result["loss"] = loss
        
        return result
    
    def get_attention_heads_importance(
        self, dataloader, device: str = "cuda"
    ) -> torch.Tensor:
        """
        Calcule l'importance de chaque tête d'attention.
        Méthode: |gradient × activation|
        
        TODO: Implémenter sur RunPod
        - Itérer sur dataloader
        - Forward + backward
        - Calculer importance = |grad · activation|
        - Agréger par tête
        
        Returns:
            Tensor de shape (num_layers, num_heads) avec scores d'importance
        """
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        importance = torch.zeros(num_layers, num_heads)
        
        # TODO: Implémenter calcul d'importance
        logger.warning("⚠️ TODO: Implement attention heads importance calculation")
        
        return importance


def load_teacher(config: Dict) -> Tuple[TeacherModel, AutoTokenizer]:
    """
    Charge le modèle teacher et son tokenizer.
    
    Args:
        config: Configuration YAML chargée
        
    Returns:
        (teacher_model, tokenizer)
    """
    teacher_config = config["teacher"]
    model_name = teacher_config["model_name"]
    
    logger.info(f"📥 Loading teacher model: {model_name}")
    teacher = TeacherModel(
        model_name=model_name,
        check_crf=teacher_config.get("check_crf", True),
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info(f"✅ Teacher loaded: {teacher.config.num_hidden_layers} layers, "
                f"{teacher.config.num_labels} labels")
    
    return teacher, tokenizer


def create_student(
    config: Dict,
    teacher: TeacherModel,
    tokenizer: AutoTokenizer,
) -> StudentModel:
    """
    Crée le modèle student à partir du teacher.
    
    Args:
        config: Configuration YAML chargée
        teacher: Modèle teacher
        tokenizer: Tokenizer (même que teacher)
        
    Returns:
        student_model
    """
    student_config = config["student"]
    
    logger.info(f"🔨 Creating student model with {student_config['num_layers']} layers")
    
    student = StudentModel(
        base_model=student_config["base_model"],
        num_layers=student_config["num_layers"],
        num_labels=teacher.config.num_labels,
        teacher_model=teacher if student_config.get("copy_embeddings") else None,
        copy_embeddings=student_config.get("copy_embeddings", True),
        copy_classifier=student_config.get("copy_classifier", True),
        copy_crf=student_config.get("copy_crf", True),
    )
    
    logger.info(f"✅ Student created: {student.config.num_hidden_layers} layers")
    
    return student


def validate_teacher(
    teacher: TeacherModel,
    tokenizer: AutoTokenizer,
    sample_texts: list,
    device: str = "cuda",
):
    """
    Valide que le teacher fonctionne correctement sur des exemples.
    
    Args:
        teacher: Modèle teacher
        tokenizer: Tokenizer
        sample_texts: Liste de phrases de test
        device: "cuda" ou "cpu"
    """
    logger.info("🔍 Validating teacher model on sample texts...")
    
    teacher.to(device)
    teacher.eval()
    
    for i, text in enumerate(sample_texts[:3]):  # Tester sur 3 exemples
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = teacher(**inputs)
        
        # Décoder prédictions
        predictions = torch.argmax(outputs["logits"], dim=-1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_labels = predictions[0].cpu().numpy()
        
        logger.info(f"  Example {i+1}: {len(tokens)} tokens, "
                    f"logits shape: {outputs['logits'].shape}")
    
    logger.info("✅ Teacher validation complete")
