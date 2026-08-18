import os
import random
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    errors = np.abs(preds - targets)
    mae = float(errors.mean())
    rmse = float(np.sqrt(((preds - targets) ** 2).mean()))
    
    # R2 Score
    if len(targets) > 1 and np.var(targets) > 1e-6:
        r2 = float(r2_score(targets, preds))
    else:
        r2 = 0.0
        
    acc_1 = float((errors <= 1.0).mean() * 100.0)
    acc_3 = float((errors <= 3.0).mean() * 100.0)
    acc_5 = float((errors <= 5.0).mean() * 100.0)
    acc_10 = float((errors <= 10.0).mean() * 100.0)
    signed_bias = float((preds - targets).mean())
    
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": signed_bias,
        "r2": r2,
        "acc_1": acc_1,
        "acc_3": acc_3,
        "acc_5": acc_5,
        "acc_10": acc_10,
    }

def compute_age_bin_metrics(preds: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
    labels = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
    
    df = pd.DataFrame({"target": targets, "pred": preds, "abs_err": np.abs(preds - targets)})
    df["age_group"] = pd.cut(df["target"], bins=bins, labels=labels, right=True)
    
    summary = df.groupby("age_group", observed=False).agg(
        count=("abs_err", "count"),
        mae=("abs_err", "mean"),
        rmse=("abs_err", lambda x: np.sqrt((x**2).mean()) if len(x) > 0 else 0),
        acc_5=("abs_err", lambda x: (x <= 5.0).mean() * 100.0 if len(x) > 0 else 0)
    ).reset_index()
    
    return summary

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, filename: str = "checkpoint.pt", best_filename: str = "best_model.pt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_path)

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

def plot_loss_curves(train_losses, val_losses, train_maes, val_maes, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-s", label="Val Loss")
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    ax2.plot(epochs, train_maes, "b-o", label="Train MAE")
    ax2.plot(epochs, val_maes, "r-s", label="Val MAE")
    ax2.set_title("Training and Validation MAE", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (Years)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_predicted_vs_true(preds, targets, save_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.3, color="#1f77b4", edgecolors="none", s=15)
    
    min_val = min(np.min(targets), np.min(preds), 0)
    max_val = max(np.max(targets), np.max(preds), 100)
    
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y = x)")
    plt.fill_between([min_val, max_val], [min_val - 5, max_val - 5], [min_val + 5, max_val + 5],
                     color="green", alpha=0.15, label="±5 Years Margin")
    
    plt.title("Predicted vs True Age", fontsize=14, fontweight="bold")
    plt.xlabel("Ground Truth Age", fontsize=12)
    plt.ylabel("Predicted Age", fontsize=12)
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_distribution(preds, targets, save_path):
    errors = preds - targets
    plt.figure(figsize=(9, 5))
    plt.hist(errors, bins=60, color="#2ca02c", edgecolor="black", alpha=0.7, density=True)
    plt.axvline(0, color="red", linestyle="--", lw=2, label="Zero Error")
    plt.axvline(np.mean(errors), color="blue", linestyle="-", lw=2, label=f"Mean Error: {np.mean(errors):.2f}")
    
    plt.title("Error Distribution (Pred - Target)", fontsize=14, fontweight="bold")
    plt.xlabel("Residual Error (Years)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_per_age_mae(preds, targets, save_path):
    bin_df = compute_age_bin_metrics(preds, targets)
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(bin_df["age_group"].astype(str), bin_df["mae"], color="#ff7f0e", edgecolor="black", alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height) and height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{height:.2f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
            
    plt.title("Per-Age-Group MAE", fontsize=14, fontweight="bold")
    plt.xlabel("Age Group", fontsize=12)
    plt.ylabel("MAE (Years)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
