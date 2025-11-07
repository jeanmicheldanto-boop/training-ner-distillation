"""
Fonctions de perte pour distillation multi-niveaux (Patient KD).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Calcule les 4 pertes de distillation:
    - L_CE: Cross-Entropy sur labels BIOES
    - L_KD: KL Divergence sur logits (soft targets)
    - L_Hidden: Cosine similarity sur hidden states appariées
    - L_CRF: L2 sur matrices de transition CRF
    """
    
    def __init__(
        self,
        temperature: float = 2.5,
        teacher_layers: List[int] = [2, 4, 6, 8, 10, 12],
        student_layers: List[int] = [2, 3, 5, 7, 9, 10],
    ):
        """
        Args:
            temperature: Température pour softening des logits (T)
            teacher_layers: Indices des couches teacher à utiliser
            student_layers: Indices des couches student correspondantes
        """
        super().__init__()
        self.temperature = temperature
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        
        # Vérifier cohérence mapping
        assert len(teacher_layers) == len(student_layers), \
            "Teacher and student layer lists must have same length"
        
        logger.info(f"📐 Distillation loss initialized:")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Layer mapping: {dict(zip(teacher_layers, student_layers))}")
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_weights: List[float] = [1.0, 1.0, 0.2, 0.2],
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule les pertes de distillation combinées.
        
        Args:
            student_outputs: Sorties du student (logits, hidden_states)
            teacher_outputs: Sorties du teacher (logits, hidden_states, crf_transitions)
            labels: Labels gold (batch, seq_len)
            attention_mask: Masque d'attention (batch, seq_len)
            loss_weights: [w_ce, w_kd, w_hidden, w_crf]
            
        Returns:
            Dict avec pertes individuelles et totale
        """
        w_ce, w_kd, w_hidden, w_crf = loss_weights
        
        # 1. Cross-Entropy Loss (L_CE)
        loss_ce = self.compute_ce_loss(
            student_outputs["logits"],
            labels,
            attention_mask,
        )
        
        # 2. KL Divergence Loss (L_KD)
        loss_kd = self.compute_kd_loss(
            student_outputs["logits"],
            teacher_outputs["logits"],
            attention_mask,
        )
        
        # 3. Hidden States Loss (L_Hidden)
        loss_hidden = self.compute_hidden_loss(
            student_outputs.get("hidden_states"),
            teacher_outputs.get("hidden_states"),
            attention_mask,
        )
        
        # 4. CRF Transitions Loss (L_CRF)
        loss_crf = self.compute_crf_loss(
            student_outputs.get("crf_transitions"),
            teacher_outputs.get("crf_transitions"),
        )
        
        # Perte totale pondérée
        total_loss = (
            w_ce * loss_ce +
            w_kd * loss_kd +
            w_hidden * loss_hidden +
            w_crf * loss_crf
        )
        
        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_hidden": loss_hidden,
            "loss_crf": loss_crf,
        }
    
    def compute_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-Entropy sur labels BIOES.
        
        Args:
            logits: (batch, seq_len, num_labels)
            labels: (batch, seq_len)
            attention_mask: (batch, seq_len)
            
        Returns:
            Scalar loss
        """
        loss_fct = nn.CrossEntropyLoss()
        
        # Ne calculer loss que sur tokens actifs (masque)
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, logits.size(-1))[active_loss]
        active_labels = labels.view(-1)[active_loss]
        
        loss = loss_fct(active_logits, active_labels)
        return loss
    
    def compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL Divergence entre distributions softened (température T).
        
        Formule: KL(P_teacher || P_student) × T²
        
        Args:
            student_logits: (batch, seq_len, num_labels)
            teacher_logits: (batch, seq_len, num_labels)
            attention_mask: (batch, seq_len)
            
        Returns:
            Scalar loss
        """
        T = self.temperature
        
        # Soften avec température
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        
        # KL Divergence
        kl_div = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="none",
        ).sum(dim=-1)  # (batch, seq_len)
        
        # Appliquer masque et moyenner
        active_mask = attention_mask.bool()
        kl_div = kl_div * active_mask
        loss = kl_div.sum() / active_mask.sum()
        
        # Multiplier par T² pour compenser temperature scaling
        loss = loss * (T ** 2)
        
        return loss
    
    def compute_hidden_loss(
        self,
        student_hidden: Optional[Tuple[torch.Tensor]],
        teacher_hidden: Optional[Tuple[torch.Tensor]],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine similarity sur hidden states appariées.
        
        Formule: L_h = (1 - cos_sim(h_s, h_t)) moyenné sur couches appariées
        
        Args:
            student_hidden: Tuple de (batch, seq_len, hidden_size)
            teacher_hidden: Tuple de (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
            
        Returns:
            Scalar loss
        """
        if student_hidden is None or teacher_hidden is None:
            return torch.tensor(0.0, device=attention_mask.device)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Itérer sur paires de couches appariées
        for t_idx, s_idx in zip(self.teacher_layers, self.student_layers):
            # Extraire hidden states (indices 0-based, +1 car hidden_states[0] = embeddings)
            h_teacher = teacher_hidden[t_idx]  # (batch, seq_len, hidden)
            h_student = student_hidden[s_idx]  # (batch, seq_len, hidden)
            
            # Cosine similarity par token
            cos_sim = F.cosine_similarity(h_student, h_teacher, dim=-1)  # (batch, seq_len)
            
            # Appliquer masque et moyenner
            active_mask = attention_mask.bool()
            cos_sim = cos_sim * active_mask
            loss_layer = (1.0 - cos_sim).sum() / active_mask.sum()
            
            total_loss += loss_layer
            num_pairs += 1
        
        # Moyenner sur toutes les paires de couches
        avg_loss = total_loss / num_pairs if num_pairs > 0 else 0.0
        
        return avg_loss
    
    def compute_crf_loss(
        self,
        student_transitions: Optional[torch.Tensor],
        teacher_transitions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        L2 loss sur matrices de transition CRF.
        
        Args:
            student_transitions: (num_labels, num_labels)
            teacher_transitions: (num_labels, num_labels)
            
        Returns:
            Scalar loss
        """
        if student_transitions is None or teacher_transitions is None:
            return torch.tensor(0.0)
        
        # L2 distance entre matrices
        loss = F.mse_loss(student_transitions, teacher_transitions)
        
        return loss


class AdaptiveWeightScheduler:
    """
    Scheduler pour ajuster les poids des pertes au fil des epochs.
    Phase 1 (warm-up): Focus sur L_CE
    Phase 2+: Full distillation
    """
    
    def __init__(
        self,
        phase1_weights: List[float],
        phase2_weights: List[float],
        phase1_epochs: int = 1,
    ):
        """
        Args:
            phase1_weights: [w_ce, w_kd, w_hidden, w_crf] pour phase 1
            phase2_weights: [w_ce, w_kd, w_hidden, w_crf] pour phase 2+
            phase1_epochs: Nombre d'epochs en phase 1
        """
        self.phase1_weights = phase1_weights
        self.phase2_weights = phase2_weights
        self.phase1_epochs = phase1_epochs
        
        logger.info(f"📊 Adaptive weight scheduler:")
        logger.info(f"  Phase 1 (epochs 1-{phase1_epochs}): {phase1_weights}")
        logger.info(f"  Phase 2 (epochs {phase1_epochs+1}+): {phase2_weights}")
    
    def get_weights(self, epoch: int) -> List[float]:
        """
        Retourne les poids pour l'epoch donné.
        
        Args:
            epoch: Numéro d'epoch (1-indexed)
            
        Returns:
            [w_ce, w_kd, w_hidden, w_crf]
        """
        if epoch <= self.phase1_epochs:
            return self.phase1_weights
        else:
            return self.phase2_weights


def create_distillation_loss(config: Dict) -> Tuple[DistillationLoss, AdaptiveWeightScheduler]:
    """
    Crée les objets de perte à partir de la config.
    
    Args:
        config: Configuration YAML chargée
        
    Returns:
        (loss_fn, weight_scheduler)
    """
    distill_config = config["distillation"]
    student_config = config["student"]
    
    # Créer fonction de perte
    loss_fn = DistillationLoss(
        temperature=distill_config["temperature"],
        teacher_layers=student_config["layer_mapping"]["teacher"],
        student_layers=student_config["layer_mapping"]["student"],
    )
    
    # Créer scheduler de poids
    weight_scheduler = AdaptiveWeightScheduler(
        phase1_weights=distill_config["loss_weights_phase1"],
        phase2_weights=distill_config["loss_weights_phase2"],
        phase1_epochs=distill_config["phase1_epochs"],
    )
    
    return loss_fn, weight_scheduler
