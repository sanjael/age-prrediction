import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config
from utils import set_seed, setup_logger, compute_metrics, save_checkpoint, plot_loss_curves
from dataset import get_dataloaders
from models import AgeModel

def get_loss_function(cfg: Config):
    loss_type = cfg.loss_type.lower()
    
    if loss_type == "huber":
        reg_loss = nn.SmoothL1Loss(beta=cfg.huber_delta)
        def loss_fn(outputs, age_target, cls_target):
            return reg_loss(outputs["pred_age"], age_target)
        return loss_fn
        
    elif loss_type == "l1":
        reg_loss = nn.L1Loss()
        def loss_fn(outputs, age_target, cls_target):
            return reg_loss(outputs["pred_age"], age_target)
        return loss_fn
        
    elif loss_type == "mse":
        reg_loss = nn.MSELoss()
        def loss_fn(outputs, age_target, cls_target):
            return reg_loss(outputs["pred_age"], age_target)
        return loss_fn
        
    elif loss_type == "ce":
        ce_loss = nn.CrossEntropyLoss()
        def loss_fn(outputs, age_target, cls_target):
            return ce_loss(outputs["logits"], cls_target)
        return loss_fn
        
    elif loss_type == "ordinal":
        bce_loss = nn.BCEWithLogitsLoss()
        def loss_fn(outputs, age_target, cls_target):
            # Target ordinal matrix: shape [B, 99]
            # binary indicators: 1 if target_age > k, else 0 for k in 1..99
            k_thresholds = torch.arange(1, 100, device=age_target.device).unsqueeze(0)  # [1, 99]
            target_matrix = (age_target.unsqueeze(1) > k_thresholds).float()  # [B, 99]
            return bce_loss(outputs["logits"], target_matrix)
        return loss_fn
        
    elif loss_type == "hybrid":
        reg_loss = nn.SmoothL1Loss(beta=cfg.huber_delta)
        ce_loss = nn.CrossEntropyLoss()
        def loss_fn(outputs, age_target, cls_target):
            l_reg = reg_loss(outputs["pred_age"], age_target)
            l_cls = ce_loss(outputs["logits"], cls_target)
            return l_reg + cfg.hybrid_ce_weight * l_cls
        return loss_fn
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
    mixed_precision: bool = True
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Train", leave=False)
    
    for step, (images, age_targets, cls_targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        age_targets = age_targets.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)
        
        with autocast(enabled=mixed_precision):
            outputs = model(images)
            loss = criterion(outputs, age_targets, cls_targets)
            loss = loss / grad_accum_steps
            
        if mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            if mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            optimizer.zero_grad()
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                try:
                    scheduler.step()
                except Exception:
                    pass
                
        total_loss += loss.item() * grad_accum_steps
        preds = outputs["pred_age"].detach().cpu().numpy()
        targets = age_targets.detach().cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(targets)
        
        current_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}", "mae": f"{current_mae:.2f}"})
        
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(loader)
    return metrics

def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
    desc: str = "Val"
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, age_targets, cls_targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            age_targets = age_targets.to(device, non_blocking=True)
            cls_targets = cls_targets.to(device, non_blocking=True)
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, age_targets, cls_targets)
                
            total_loss += loss.item()
            preds = outputs["pred_age"].detach().cpu().numpy()
            targets = age_targets.detach().cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(loader)
    return metrics, all_preds, all_targets

def run_training(cfg: Config, exp_name: str = "baseline"):
    set_seed(cfg.seed)
    
    # Setup paths and logger
    exp_dir = os.path.join(cfg.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logger(exp_name, os.path.join(exp_dir, "train.log"))
    
    logger.info("=" * 60)
    logger.info(f"STARTING EXPERIMENT: {exp_name}")
    logger.info(f"Backbone: {cfg.backbone} | Head: {cfg.head_type} | Loss: {cfg.loss_type}")
    logger.info(f"Batch Size: {cfg.batch_size} (Accum: {cfg.gradient_accumulation_steps}) | Aug: {cfg.aug_level}")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. DataLoaders
    logger.info("Loading dataset splits...")
    dataloaders = get_dataloaders(cfg)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    
    logger.info(f"Train samples: {len(dataloaders['train_df']):,} | Val: {len(dataloaders['val_df']):,} | Test: {len(dataloaders['test_df']):,}")
    
    # 2. Model Initialization
    logger.info(f"Initializing model: {cfg.backbone} with {cfg.head_type} head...")
    model = AgeModel(
        backbone_name=cfg.backbone,
        head_type=cfg.head_type,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes
    ).to(device)
    
    # 3. Loss & Optimizer
    criterion = get_loss_function(cfg)
    
    # Checkpoint Resuming logic
    start_epoch = 1
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    
    if cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        logger.info(f"Loading checkpoint from: {cfg.resume_checkpoint}")
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            ckpt_epoch = ckpt.get("epoch", 0)
            start_epoch = cfg.start_epoch if cfg.start_epoch > 1 else ckpt_epoch + 1
            if "best_val_mae" in ckpt and ckpt["best_val_mae"] is not None:
                best_val_mae = ckpt["best_val_mae"]
            if "history" in ckpt and isinstance(ckpt["history"], list):
                history = list(ckpt["history"])
                for h in ckpt["history"]:
                    if h.get("val_mae") and h["val_mae"] < best_val_mae:
                        best_val_mae = h["val_mae"]
                        best_epoch = h.get("epoch", 0)
            logger.info(f"Successfully resumed! Starting from Epoch {start_epoch:02d} to {cfg.epochs:02d} (Previous Best Val MAE: {best_val_mae:.2f} yrs)")
        else:
            model.load_state_dict(ckpt)
            start_epoch = cfg.start_epoch
            logger.info(f"Loaded model weights. Starting from Epoch {start_epoch:02d} to {cfg.epochs:02d}")
            
    param_groups = model.get_parameter_groups(head_lr=cfg.lr, backbone_lr=cfg.backbone_lr, weight_decay=cfg.weight_decay)
    
    if cfg.optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(param_groups)
        
    steps_per_epoch = (len(train_loader) + cfg.gradient_accumulation_steps - 1) // cfg.gradient_accumulation_steps
    epochs_to_train = max(1, cfg.epochs - start_epoch + 1)
    total_steps = steps_per_epoch * epochs_to_train
    
    if cfg.scheduler_type.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=total_steps,
            pct_start=0.15,
            anneal_strategy="cos"
        )
    elif cfg.scheduler_type.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs_to_train,
            eta_min=cfg.min_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
    scaler = GradScaler(enabled=cfg.mixed_precision and torch.cuda.is_available())
    
    # Multi-stage stage 1 handling
    if cfg.multi_stage:
        logger.info(f"Stage 1 active: Freezing backbone for first {cfg.stage1_freeze_epochs} epochs.")
        model.freeze_backbone(True)
        
    start_time = time.time()
    
    # 4. Training Loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()
        
        # Multi-stage unfreezing logic
        if cfg.multi_stage:
            if epoch == cfg.stage1_freeze_epochs + 1:
                logger.info(f"Stage 2 active: Unfreezing last layers of backbone at epoch {epoch}...")
                model.unfreeze_last_n_blocks(n=2)
            elif epoch == (cfg.stage1_freeze_epochs * 2) + 1:
                logger.info(f"Stage 3 active: Unfreezing entire model for full fine-tuning at epoch {epoch}...")
                model.freeze_backbone(False)
                
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler if cfg.scheduler_type.lower() == "onecycle" else None,
            scaler=scaler,
            device=device,
            grad_accum_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision
        )
        
        # Validation epoch
        val_metrics, _, _ = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc="Val"
        )
        
        if cfg.scheduler_type.lower() != "onecycle" and scheduler is not None:
            scheduler.step()
            
        epoch_time = time.time() - epoch_start
        
        # Check if best model
        is_best = val_metrics["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Logging
        logger.info(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}, R2: {train_metrics['r2']:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.2f}, Acc@5: {val_metrics['acc_5']:.1f}% "
            f"{'[*] BEST' if is_best else ''}"
        )
        
        # Record history
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_acc_5": val_metrics["acc_5"],
            "is_best": is_best
        }
        history.append(record)
        
        # Save Checkpoint
        state = {
            "epoch": epoch,
            "config": cfg.__dict__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae": best_val_mae,
            "history": history
        }
        save_checkpoint(
            state=state,
            is_best=is_best,
            checkpoint_dir=exp_dir,
            filename=f"checkpoint_epoch_{epoch:02d}.pt",
            best_filename="best_model.pt"
        )
        
        # Early Stopping check
        if patience_counter >= cfg.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {cfg.early_stopping_patience} epochs).")
            break
            
    total_training_time = time.time() - start_time
    logger.info(f"Training finished in {total_training_time/60:.1f} minutes. Best Epoch: {best_epoch} with Val MAE: {best_val_mae:.2f}")
    
    # Save History CSV and Plots
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(exp_dir, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)
    
    plot_loss_curves(
        train_losses=history_df["train_loss"].values,
        val_losses=history_df["val_loss"].values,
        train_maes=history_df["train_mae"].values,
        val_maes=history_df["val_mae"].values,
        save_path=os.path.join(exp_dir, "loss_curves.png")
    )
    
    # 5. Final Evaluation on Held-Out Test Set (Only if explicitly unlocked)
    if not cfg.lock_test:
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING BEST MODEL ON HELD-OUT TEST SPLIT...")
        logger.info("=" * 60)
        
        best_ckpt_path = os.path.join(exp_dir, "best_model.pt")
        if os.path.exists(best_ckpt_path):
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
        test_metrics, test_preds, test_targets = evaluate_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            desc="Test"
        )
        
        logger.info(
            f"TEST RESULTS: "
            f"MAE: {test_metrics['mae']:.2f} yrs | "
            f"RMSE: {test_metrics['rmse']:.2f} yrs | "
            f"Bias: {test_metrics['bias']:+.2f} yrs | "
            f"R2: {test_metrics['r2']:.4f} | "
            f"Acc@+/-1yr: {test_metrics['acc_1']:.2f}% | "
            f"Acc@+/-3yr: {test_metrics['acc_3']:.2f}% | "
            f"Acc@+/-5yr: {test_metrics['acc_5']:.2f}% | "
            f"Acc@+/-10yr: {test_metrics['acc_10']:.2f}%"
        )
    else:
        logger.info("\n" + "=" * 60)
        logger.info("[LOCKED TEST RULE] Test evaluation bypassed. Model selection strictly on Val MAE.")
        logger.info(f"FINAL BEST VALIDATION MAE: {best_val_mae:.2f} yrs (Epoch {best_epoch:02d})")
        logger.info("=" * 60)
        test_metrics = {"mae": None, "rmse": None, "bias": None, "r2": None, "acc_5": None}
    
    return {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_acc_5": test_metrics["acc_5"],
        "exp_dir": exp_dir
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Facial Age Estimation Model")
    parser.add_argument("--backbone", type=str, default="tf_efficientnetv2_s", help="Backbone model")
    parser.add_argument("--head", type=str, default="direct", choices=["direct", "dex", "ordinal", "hybrid"], help="Head type")
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "l1", "mse", "ce", "ordinal", "hybrid"], help="Loss function")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Head learning rate")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="Backbone learning rate")
    parser.add_argument("--aug", type=str, default="moderate", choices=["minimal", "moderate", "strong"], help="Augmentation level")
    parser.add_argument("--weighted_sampler", action="store_true", help="Use weighted sampler")
    parser.add_argument("--multi_stage", action="store_true", help="Use multi-stage unfreezing")
    parser.add_argument("--manifest", type=str, default=None, help="Custom manifest path for preprocessed crops")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution")
    parser.add_argument("--unlock_test", action="store_true", help="Unlock test set for final exam only")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--start_epoch", type=int, default=1, help="Starting epoch number when resuming")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience epochs")
    parser.add_argument("--exp_name", type=str, default="baseline_effnetv2_s", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    cfg = Config(
        backbone=args.backbone,
        head_type=args.head,
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        aug_level=args.aug,
        use_weighted_sampler=args.weighted_sampler,
        multi_stage=args.multi_stage,
        img_size=args.img_size,
        lock_test=not args.unlock_test,
        resume_checkpoint=args.resume_checkpoint,
        start_epoch=args.start_epoch,
        early_stopping_patience=args.patience,
        seed=args.seed
    )
    if args.manifest:
        cfg.manifest_path = args.manifest
        
    run_training(cfg, exp_name=args.exp_name)
