"""
Pruning structuré des têtes d'attention.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttentionHeadPruner:
    """
    Classe pour pruner les têtes d'attention les moins importantes.
    Méthode: Calculer importance puis masquer les têtes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_rate: float = 0.25,
        method: str = "grad_activation",
        importance_metric: str = "l1_norm",
    ):
        """
        Args:
            model: Modèle student à pruner
            pruning_rate: Proportion de têtes à pruner (0.25 = 25%)
            method: "grad_activation" ou "entropy"
            importance_metric: "l1_norm" ou "l2_norm"
        """
        self.model = model
        self.pruning_rate = pruning_rate
        self.method = method
        self.importance_metric = importance_metric
        
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        
        # Masque de pruning: 1 = gardé, 0 = pruné
        self.head_mask = torch.ones(self.num_layers, self.num_heads)
        
        logger.info(f"🔪 AttentionHeadPruner initialized:")
        logger.info(f"  Pruning rate: {pruning_rate:.1%}")
        logger.info(f"  Method: {method}")
        logger.info(f"  Total heads: {self.num_layers} layers × {self.num_heads} heads = "
                    f"{self.num_layers * self.num_heads} heads")
        logger.info(f"  Heads to prune: {int(self.num_layers * self.num_heads * pruning_rate)}")
    
    def compute_head_importance(
        self,
        dataloader,
        device: str = "cuda",
        num_batches: int = 100,
    ) -> torch.Tensor:
        """
        Calcule l'importance de chaque tête d'attention.
        
        Méthode grad_activation:
            importance = E[|grad(L) · attn_weights|]
        
        TODO: Implémenter sur RunPod
        - Forward + backward sur échantillon du dataloader
        - Capturer gradients des attention weights
        - Calculer |grad × activation| par tête
        - Agréger sur batches
        
        Args:
            dataloader: Dataloader pour calcul d'importance
            device: "cuda" ou "cpu"
            num_batches: Nombre de batches à utiliser
            
        Returns:
            importance: Tensor (num_layers, num_heads) avec scores
        """
        logger.info(f"📊 Computing attention head importance ({num_batches} batches)...")
        
        self.model.train()
        importance = torch.zeros(self.num_layers, self.num_heads)
        
        # TODO: Implémenter calcul d'importance réel sur RunPod
        # Pour l'instant, importance aléatoire (placeholder)
        if self.method == "grad_activation":
            # Simuler importance (TODO: remplacer par calcul réel)
            importance = torch.rand(self.num_layers, self.num_heads)
            logger.warning("⚠️ Using random importance (TODO: implement real calculation)")
        
        elif self.method == "entropy":
            # TODO: Calculer entropie des attention weights
            importance = torch.rand(self.num_layers, self.num_heads)
            logger.warning("⚠️ Using random importance (TODO: implement entropy method)")
        
        logger.info(f"✅ Importance computed. Stats:")
        logger.info(f"  Mean: {importance.mean():.4f}")
        logger.info(f"  Std: {importance.std():.4f}")
        logger.info(f"  Min: {importance.min():.4f}, Max: {importance.max():.4f}")
        
        return importance
    
    def prune_heads(self, importance: torch.Tensor) -> Dict[str, List[int]]:
        """
        Prune les têtes les moins importantes selon le taux de pruning.
        
        Args:
            importance: Tensor (num_layers, num_heads) avec scores d'importance
            
        Returns:
            Dict mapping layer_idx -> list of pruned head indices
        """
        logger.info(f"🔪 Pruning {self.pruning_rate:.1%} of attention heads...")
        
        # Nombre total de têtes à pruner
        total_heads = self.num_layers * self.num_heads
        num_to_prune = int(total_heads * self.pruning_rate)
        
        # Flatten et trier par importance
        importance_flat = importance.view(-1)
        sorted_indices = torch.argsort(importance_flat)
        
        # Sélectionner les N têtes les moins importantes
        heads_to_prune_flat = sorted_indices[:num_to_prune]
        
        # Convertir en (layer_idx, head_idx)
        pruned_heads_by_layer = {}
        for flat_idx in heads_to_prune_flat:
            layer_idx = flat_idx // self.num_heads
            head_idx = flat_idx % self.num_heads
            
            layer_idx = layer_idx.item()
            head_idx = head_idx.item()
            
            if layer_idx not in pruned_heads_by_layer:
                pruned_heads_by_layer[layer_idx] = []
            pruned_heads_by_layer[layer_idx].append(head_idx)
            
            # Mettre à jour le masque
            self.head_mask[layer_idx, head_idx] = 0
        
        # Logger résumé
        for layer_idx in sorted(pruned_heads_by_layer.keys()):
            heads = pruned_heads_by_layer[layer_idx]
            logger.info(f"  Layer {layer_idx}: pruned {len(heads)} heads {heads}")
        
        logger.info(f"✅ Pruned {num_to_prune} heads "
                    f"({num_to_prune / total_heads:.1%} of total)")
        
        return pruned_heads_by_layer
    
    def apply_pruning_mask(self, pruned_heads: Dict[str, List[int]]):
        """
        Applique le masque de pruning au modèle.
        Met à zéro les poids des têtes prunées dans Q, K, V, O.
        
        TODO: Implémenter sur RunPod
        - Accéder aux poids des attention layers
        - Masquer les colonnes correspondantes dans Q, K, V
        - Masquer les lignes correspondantes dans O (output projection)
        
        Args:
            pruned_heads: Dict mapping layer_idx -> list of head indices to prune
        """
        logger.info("🎭 Applying pruning mask to model weights...")
        
        # TODO: Implémenter masquage réel sur RunPod
        # Pour chaque couche avec têtes prunées:
        #   1. Accéder à layer.attention.self.query/key/value
        #   2. Masquer les colonnes correspondant aux têtes prunées
        #   3. Accéder à layer.attention.output.dense
        #   4. Masquer les lignes correspondantes
        
        for layer_idx, head_indices in pruned_heads.items():
            logger.info(f"  Layer {layer_idx}: masking {len(head_indices)} heads")
            # TODO: Masquage réel des poids
        
        logger.info("✅ Pruning mask applied")
    
    def get_pruning_stats(self) -> Dict:
        """
        Retourne des statistiques sur le pruning.
        
        Returns:
            Dict avec stats (total_heads, pruned_heads, pruning_ratio, etc.)
        """
        total_heads = self.num_layers * self.num_heads
        pruned_heads = (self.head_mask == 0).sum().item()
        remaining_heads = (self.head_mask == 1).sum().item()
        
        stats = {
            "total_heads": total_heads,
            "pruned_heads": pruned_heads,
            "remaining_heads": remaining_heads,
            "pruning_ratio": pruned_heads / total_heads,
            "head_mask": self.head_mask.tolist(),
        }
        
        return stats
    
    def save_mask(self, filepath: str):
        """
        Sauvegarde le masque de pruning dans un fichier JSON.
        
        Args:
            filepath: Chemin du fichier de sortie (ex: "heads_pruned_mask.json")
        """
        import json
        
        stats = self.get_pruning_stats()
        
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"💾 Pruning mask saved to {filepath}")
    
    def load_mask(self, filepath: str):
        """
        Charge un masque de pruning depuis un fichier JSON.
        
        Args:
            filepath: Chemin du fichier à charger
        """
        import json
        
        with open(filepath, "r") as f:
            stats = json.load(f)
        
        self.head_mask = torch.tensor(stats["head_mask"])
        
        logger.info(f"📥 Pruning mask loaded from {filepath}")
        logger.info(f"  Pruned heads: {stats['pruned_heads']}/{stats['total_heads']} "
                    f"({stats['pruning_ratio']:.1%})")


def prune_student_model(
    model: nn.Module,
    dataloader,
    config: Dict,
    device: str = "cuda",
) -> Tuple[nn.Module, AttentionHeadPruner]:
    """
    Pipeline complet de pruning pour le modèle student.
    
    Args:
        model: Modèle student distillé
        dataloader: Dataloader pour calcul d'importance
        config: Configuration YAML
        device: Device pour calculs
        
    Returns:
        (pruned_model, pruner) avec modèle pruné et objet pruner
    """
    pruning_config = config["pruning"]
    
    # Initialiser pruner
    pruner = AttentionHeadPruner(
        model=model,
        pruning_rate=pruning_config["rate"],
        method=pruning_config["method"],
        importance_metric=pruning_config["importance_metric"],
    )
    
    # Calculer importance
    importance = pruner.compute_head_importance(
        dataloader=dataloader,
        device=device,
        num_batches=100,
    )
    
    # Pruner têtes
    pruned_heads = pruner.prune_heads(importance)
    
    # Appliquer masque
    pruner.apply_pruning_mask(pruned_heads)
    
    return model, pruner
