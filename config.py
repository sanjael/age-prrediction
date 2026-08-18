import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    # --- Paths ---
    project_root: str = r"e:\CTS FINAL\DATA SET"
    data_dir: str = r"e:\CTS FINAL\DATA SET\age_prediction_up\age_prediction"
    train_dir: str = r"e:\CTS FINAL\DATA SET\age_prediction_up\age_prediction\train"
    test_dir: str = r"e:\CTS FINAL\DATA SET\age_prediction_up\age_prediction\test"
    
    manifest_path: str = r"e:\CTS FINAL\DATA SET\manifest_clean.csv"
    output_dir: str = r"e:\CTS FINAL\DATA SET\outputs"
    checkpoint_dir: str = r"e:\CTS FINAL\DATA SET\outputs\checkpoints"
    log_dir: str = r"e:\CTS FINAL\DATA SET\outputs\logs"
    plots_dir: str = r"e:\CTS FINAL\DATA SET\outputs\plots"
    
    # --- Model Configuration ---
    backbone: str = "tf_efficientnetv2_s"  # tf_efficientnetv2_s, resnet50, convnext_small, efficientnet_b3
    head_type: str = "direct"  # "direct", "dex", "ordinal", "hybrid"
    pretrained: bool = True
    num_classes: int = 100  # For DEX head: ages 1 to 100
    min_age: int = 1
    max_age: int = 100
    img_size: int = 224
    
    # --- Training Hyperparameters ---
    epochs: int = 50
    batch_size: int = 16  # Tuned for RTX 3050 Ti 4GB VRAM
    gradient_accumulation_steps: int = 2  # Effective batch size = 32
    num_workers: int = 4
    
    lr: float = 1e-3  # Head LR
    backbone_lr: float = 1e-4  # Backbone differential LR
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    optimizer_type: str = "adamw"  # "adamw", "sgd"
    scheduler_type: str = "onecycle"  # "onecycle", "cosine", "step"
    warmup_epochs: int = 3
    
    # --- Loss Configuration ---
    loss_type: str = "huber"  # "huber", "l1", "mse", "ce", "ordinal", "hybrid"
    huber_delta: float = 1.0
    hybrid_ce_weight: float = 0.3
    
    # --- Sampling & Augmentation ---
    use_weighted_sampler: bool = False
    sampler_power: float = 0.5  # inverse sqrt frequency
    aug_level: str = "moderate"  # "minimal", "moderate", "strong"
    
    # --- Training Stages ---
    multi_stage: bool = False
    stage1_freeze_epochs: int = 3
    
    # --- Diagnostics & Early Stopping ---
    early_stopping_patience: int = 10
    mixed_precision: bool = True
    lock_test: bool = True  # Strict rule: test set is locked, validate only on val split
    resume_checkpoint: Optional[str] = None
    start_epoch: int = 1
    seed: int = 42
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

