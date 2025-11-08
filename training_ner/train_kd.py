"""
Script principal d'entraînement par distillation (Knowledge Distillation).

Usage:
    python train_kd.py --config configs/kd_camembert.yaml --output artifacts/student_10L
"""
import argparse
import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging

from training_ner.models import load_teacher, create_student, validate_teacher
from training_ner.losses import create_distillation_loss
from training_ner.data_loader import load_label_mapping, create_dataloaders
from training_ner.utils import (
    setup_logging,
    load_config,
    set_seed,
    count_parameters,
    format_number,
    save_model,
    TrainingMonitor,
    get_device,
    format_time,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Train NER student via Knowledge Distillation")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kd_camembert.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/student_10L",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    
    return parser.parse_args()


def train_epoch(
    student,
    teacher,
    train_loader,
    optimizer,
    loss_fn,
    loss_weights,
    device,
    use_amp=True,
    grad_clip=1.0,
):
    """
    Entraîne le student pour une epoch.
    
    TODO: Implémenter sur RunPod
    - Boucle sur train_loader
    - Forward teacher + student
    - Calculer pertes distillation
    - Backward + optimizer step
    - Logger métriques
    
    Args:
        student: Modèle student
        teacher: Modèle teacher (frozen)
        train_loader: DataLoader d'entraînement
        optimizer: Optimizer
        loss_fn: Fonction de perte
        loss_weights: Poids des pertes [w_ce, w_kd, w_hidden, w_crf]
        device: Device ("cuda" ou "cpu")
        use_amp: Utiliser mixed precision (FP16)
        grad_clip: Max norm pour gradient clipping
        
    Returns:
        Dict avec métriques moyennes de l'epoch
    """
    student.train()
    teacher.eval()
    
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_kd = 0.0
    total_loss_hidden = 0.0
    total_loss_crf = 0.0
    num_batches = 0
    
    scaler = GradScaler() if use_amp else None
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        # Déplacer batch sur device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        # TODO: Implémenter forward passes
        # 1. Forward teacher (no grad)
        # 2. Forward student (with grad)
        # 3. Calculer pertes via loss_fn
        # 4. Backward + optimizer step
        # 5. Gradient clipping
        
        # Placeholder loss (TODO: remplacer)
        loss = torch.tensor(0.5, device=device, requires_grad=True)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
        
        # Accumuler métriques
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Moyennes
    metrics = {
        "loss": total_loss / num_batches,
        "loss_ce": total_loss_ce / num_batches if total_loss_ce > 0 else 0.0,
        "loss_kd": total_loss_kd / num_batches if total_loss_kd > 0 else 0.0,
        "loss_hidden": total_loss_hidden / num_batches if total_loss_hidden > 0 else 0.0,
        "loss_crf": total_loss_crf / num_batches if total_loss_crf > 0 else 0.0,
    }
    
    return metrics


def validate(student, val_loader, loss_fn, loss_weights, device):
    """
    Valide le student sur le validation set.
    
    TODO: Implémenter sur RunPod
    - Boucle sur val_loader (no grad)
    - Forward student
    - Calculer loss
    - Logger métriques
    
    Args:
        student: Modèle student
        val_loader: DataLoader de validation
        loss_fn: Fonction de perte
        loss_weights: Poids des pertes
        device: Device
        
    Returns:
        Dict avec métriques de validation
    """
    student.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # TODO: Forward + calculer loss
            loss = torch.tensor(0.5, device=device)
            
            total_loss += loss.item()
            num_batches += 1
    
    metrics = {
        "val_loss": total_loss / num_batches,
    }
    
    return metrics


def main():
    """Fonction principale d'entraînement."""
    args = parse_args()
    
    # Setup
    setup_logging()
    logger.info("=" * 80)
    logger.info("🚀 Starting Knowledge Distillation Training")
    logger.info("=" * 80)
    
    # Charger config
    config = load_config(args.config)
    logger.info(f"📋 Config loaded from {args.config}")
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Device
    device = get_device()
    
    # Charger label mapping
    label2id, id2label = load_label_mapping(config["data"]["label2id_file"])
    
    # Charger teacher
    logger.info("\n" + "=" * 80)
    logger.info("📥 LOADING TEACHER MODEL")
    logger.info("=" * 80)
    teacher, tokenizer = load_teacher(config)
    teacher.to(device)
    
    # Valider teacher sur exemples
    sample_texts = [
        "Jean-Baptiste Durand habite à Paris.",
        "Le docteur Martin travaille à l'hôpital de Lyon.",
        "Marie consulte le psychiatre Dr. Dupont à Marseille.",
    ]
    validate_teacher(teacher, tokenizer, sample_texts, device)
    
    # Créer student
    logger.info("\n" + "=" * 80)
    logger.info("🔨 CREATING STUDENT MODEL")
    logger.info("=" * 80)
    student = create_student(config, teacher, tokenizer)
    student.to(device)
    
    # Afficher taille des modèles
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    logger.info(f"\n📊 Model sizes:")
    logger.info(f"  Teacher: {format_number(teacher_params['total'])} parameters")
    logger.info(f"  Student: {format_number(student_params['total'])} parameters")
    logger.info(f"  Compression ratio: {teacher_params['total'] / student_params['total']:.2f}x")
    
    # Créer dataloaders
    logger.info("\n" + "=" * 80)
    logger.info("📁 LOADING DATA")
    logger.info("=" * 80)
    train_loader, val_loader, test_loader = create_dataloaders(config, tokenizer, label2id)
    
    # Créer loss et weight scheduler
    loss_fn, weight_scheduler = create_distillation_loss(config)
    
    # Créer optimizer et scheduler
    training_config = config["training"]
    optimizer = optim.AdamW(
        student.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    
    # TODO: Ajouter learning rate scheduler
    
    # Monitoring
    monitor = TrainingMonitor(args.output)
    
    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("🎯 STARTING TRAINING")
    logger.info("=" * 80)
    
    num_epochs = training_config["max_epochs"]
    use_amp = training_config.get("mixed_precision", True) and device == "cuda"
    
    best_val_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"📅 Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*80}")
        
        # Récupérer poids des pertes pour cette epoch
        loss_weights = weight_scheduler.get_weights(epoch)
        logger.info(f"Loss weights: {loss_weights}")
        
        # Entraînement
        epoch_start = time.time()
        train_metrics = train_epoch(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            device=device,
            use_amp=use_amp,
            grad_clip=training_config["gradient_clip"],
        )
        epoch_time = time.time() - epoch_start
        
        logger.info(f"✅ Training complete in {format_time(epoch_time)}")
        logger.info(f"  Loss: {train_metrics['loss']:.4f}")
        
        # Validation
        val_metrics = validate(
            student=student,
            val_loader=val_loader,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            device=device,
        )
        
        logger.info(f"✅ Validation complete")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Logger métriques
        all_metrics = {**train_metrics, **val_metrics, "epoch_time": epoch_time}
        monitor.log(epoch=epoch, step=epoch, metrics=all_metrics)
        
        # Sauvegarder best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            logger.info(f"💾 New best validation loss: {best_val_loss:.4f}")
            save_model(student, tokenizer, args.output, config)
    
    # Fin d'entraînement
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time: {format_time(total_time)}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {args.output}")
    
    # Sauvegarder résumé
    monitor.save_summary()


if __name__ == "__main__":
    main()
