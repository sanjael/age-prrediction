"""
train_moderate_balanced_champion.py
Moderate Age-Aware Weighted Sampler Fine-Tuning from 4.499 Champion
- Full original dataset preserved (170,030 training images)
- No physical dataset truncation / no artificial extreme oversampling
- Controlled demographic exposure via WeightedRandomSampler:
    1-12: ~7% | 13-19: ~8% | 20-35: ~28% | 36-45: ~20% | 46-60: ~17% | 61-75: ~13% | 76-100: ~7%
- 2-Stage Low-LR Fine-Tuning: Stage 1 (Head Polish, 3 ep) -> Stage 2 (Gentle Joint, 5 ep)
- Live tqdm progress bar & full demographic validation breakdown per epoch
"""
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from typing import Dict

from models import AgeModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(img_size: int = 320):
    return T.Compose([
        T.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_eval_transforms(img_size: int = 320):
    tf_orig = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    tf_flip = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return tf_orig, tf_flip

class FastAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df["filepath"].values
        self.ages = self.df["age"].values.astype(np.float32)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        age = self.ages[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (320, 320), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(age, dtype=torch.float32)

def generate_gaussian_labels(ages: torch.Tensor, num_classes: int = 100, sigma: float = 1.5, device="cuda") -> torch.Tensor:
    bins = torch.arange(1, num_classes + 1, dtype=torch.float32, device=device).unsqueeze(0)
    ages = ages.unsqueeze(1).to(device)
    dist_sq = (bins - ages) ** 2
    probs = torch.exp(-dist_sq / (2.0 * (sigma ** 2)))
    return probs / torch.sum(probs, dim=-1, keepdim=True)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = np.abs(y_true - y_pred)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    acc_1 = float(np.mean(errors <= 1.0) * 100.0)
    acc_3 = float(np.mean(errors <= 3.0) * 100.0)
    acc_5 = float(np.mean(errors <= 5.0) * 100.0)
    acc_10 = float(np.mean(errors <= 10.0) * 100.0)
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "acc_1": round(acc_1, 2),
        "acc_3": round(acc_3, 2),
        "acc_5": round(acc_5, 2),
        "acc_10": round(acc_10, 2)
    }

def print_age_breakdown(y_true: np.ndarray, y_pred: np.ndarray, title: str = "VALIDATION BREAKDOWN"):
    df = pd.DataFrame({"target": y_true, "pred": y_pred})
    df["error"] = np.abs(df["pred"] - df["target"])
    df["acc_3"] = df["error"] <= 3.0
    df["acc_5"] = df["error"] <= 5.0
    
    bins = [0, 12, 19, 35, 45, 60, 75, 105]
    labels = ['1-12 (Kids)', '13-19 (Teens)', '20-35 (Young Adults)', '36-45 (Adults)', '46-60 (Middle Age)', '61-75 (Seniors)', '76-100 (Elderly)']
    df["bracket"] = pd.cut(df["target"], bins=bins, labels=labels)
    
    res = df.groupby("bracket", observed=False).agg(
        Count=("error", "count"),
        MAE=("error", "mean"),
        RMSE=("error", lambda x: np.sqrt(np.mean(x**2))),
        Acc_3=("acc_3", lambda x: np.mean(x) * 100.0),
        Acc_5=("acc_5", lambda x: np.mean(x) * 100.0)
    ).round(3)
    
    print(f"\n--- {title} ---", flush=True)
    print(res.to_string(), flush=True)
    print("-" * 75 + "\n", flush=True)

def compute_sample_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Computes smooth, moderate demographic weights for WeightedRandomSampler.
    Target Exposure:
      1-12: 7% | 13-19: 8% | 20-35: 28% | 36-45: 20% | 46-60: 17% | 61-75: 13% | 76-100: 7%
    """
    target_probs = {
        '1-12': 0.07,
        '13-19': 0.08,
        '20-35': 0.28,
        '36-45': 0.20,
        '46-60': 0.17,
        '61-75': 0.13,
        '76-100': 0.07
    }
    bins = [0, 12, 19, 35, 45, 60, 75, 105]
    labels = ['1-12', '13-19', '20-35', '36-45', '46-60', '61-75', '76-100']
    df['bracket'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    counts = df['bracket'].value_counts()
    weight_map = {}
    for b in labels:
        c = counts[b]
        t = target_probs[b]
        weight_map[b] = t / float(c) if c > 0 else 0.0
        
    weights = df['bracket'].map(weight_map).values.astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Moderate Demographic Age-Aware Sampler Fine-Tuning")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="Stage 1 Head-Only Epochs (default: 3)")
    parser.add_argument("--stage2_epochs", type=int, default=5, help="Stage 2 Joint Fine-Tuning Epochs (default: 5)")
    parser.add_argument("--samples_per_epoch", type=int, default=20000, help="Samples drawn per epoch (default: 20,000)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps (effective batch: 32)")
    args = parser.parse_args()

    out_dir = "outputs/exp30_champion_moderate_balanced"
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85, flush=True)
    print(" [*] MODERATE DEMOGRAPHIC AGE-AWARE SAMPLER FINE-TUNING", flush=True)
    print(" [*] Anchor: Champion 4.499 Model | Full Dataset: 170,030 images | Target: Moderate Balanced Exposure", flush=True)
    print(f" [*] Device: {device} | Output Directory: {out_dir}", flush=True)
    print("=" * 85, flush=True)
    
    # 1. Load Full Master Manifest
    manifest_path = "manifest_p2_320_plus_utkface.csv"
    df = pd.read_csv(manifest_path)
    train_full = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    
    print(f"[*] Total Master Training Pool: {len(train_full):,} images (100% of all ages preserved)", flush=True)
    print(f"[*] Benchmark Validation Set  : {len(val_df):,} images (All 1-100 ages untouched)\n", flush=True)
    
    # 2. Build WeightedRandomSampler (Controlled Exposure)
    sample_weights = compute_sample_weights(train_full)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=args.samples_per_epoch, replacement=True)
    
    tf_train = get_train_transforms(320)
    tf_orig, tf_flip = get_eval_transforms(320)
    
    ds_train = FastAgeDataset(train_full, transform=tf_train)
    ds_val_orig = FastAgeDataset(val_df, transform=tf_orig)
    ds_val_flip = FastAgeDataset(val_df, transform=tf_flip)
    
    val_loader_orig = DataLoader(ds_val_orig, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    val_loader_flip = DataLoader(ds_val_flip, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Load Champion 4.499 Weights
    pretrained_path = "outputs/exp25_effnetv2s_dex_expected_age/best_model.pt"
    print(f"[*] Initializing Champion weights from: {pretrained_path}...", flush=True)
    model = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="dex", pretrained=False).to(device)
    ckpt = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("[+] Champion weights loaded successfully!\n", flush=True)
    
    # 4. Pre-Training Benchmark Baseline Check
    print("=" * 85, flush=True)
    print(" [*] COMPUTING PRE-TRAINING BENCHMARK BASELINE ON FULL VALIDATION SET...", flush=True)
    print("=" * 85, flush=True)
    model.eval()
    init_preds, init_targets = [], []
    with torch.no_grad():
        for (img_o, ys), (img_f, _) in tqdm(zip(val_loader_orig, val_loader_flip), total=len(val_loader_orig), desc="Baseline Validation"):
            img_o = img_o.to(device, non_blocking=True)
            img_f = img_f.to(device, non_blocking=True)
            p = 0.5 * model(img_o)["pred_age"].cpu().numpy() + 0.5 * model(img_f)["pred_age"].cpu().numpy()
            init_preds.extend(p.tolist())
            init_targets.extend(ys.numpy().tolist())
            
    m_init = compute_metrics(np.array(init_targets), np.array(init_preds))
    print(f"\n[+] BASELINE METRICS: MAE = {m_init['mae']:.3f} yrs | RMSE = {m_init['rmse']:.3f} | Acc@+-3 = {m_init['acc_3']}% | Acc@+-5 = {m_init['acc_5']}%", flush=True)
    print_age_breakdown(np.array(init_targets), np.array(init_preds), title="PRE-TRAINING BASELINE DEMOGRAPHIC BREAKDOWN")
    
    best_val_mae = m_init["mae"]
    best_epoch = 0
    save_checkpoint_path = os.path.join(out_dir, "best_model.pt")
    
    scaler = torch.cuda.amp.GradScaler()
    total_epochs = args.stage1_epochs + args.stage2_epochs
    history = []
    
    # =========================================================================
    # STAGE 1: Head-Only Low-LR Calibration (3 Epochs)
    # =========================================================================
    print("=" * 85, flush=True)
    print(f" [*] STAGE 1: HEAD-ONLY POLISH ({args.stage1_epochs} Epochs | Backbone Frozen)", flush=True)
    print("=" * 85, flush=True)
    
    for param in model.backbone.parameters():
        param.requires_grad = False
    optimizer_s1 = torch.optim.AdamW(model.head.parameters(), lr=1.0e-4, weight_decay=1e-4)
    
    for epoch in range(1, args.stage1_epochs + 1):
        print(f"\n{'='*30} STAGE 1 - EPOCH {epoch:02d}/{args.stage1_epochs:02d} {'='*30}", flush=True)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
        model.train()
        model.backbone.eval()
        total_loss = 0.0
        train_preds, train_targets = [], []
        epoch_start = time.time()
        optimizer_s1.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Stage 1 Ep {epoch:02d}", unit="batch")
        for step, (images, ages) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            ages = ages.to(device, non_blocking=True)
            target_gaussian = generate_gaussian_labels(ages, num_classes=100, sigma=1.5, device=device)
            
            with torch.cuda.amp.autocast():
                out = model(images)
                logits = out["logits"]
                pred_age = out["pred_age"]
                loss = -(target_gaussian * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                loss = loss / args.accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer_s1)
                scaler.update()
                optimizer_s1.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * args.accum_steps
            train_preds.extend(pred_age.detach().cpu().numpy().tolist())
            train_targets.extend(ages.cpu().numpy().tolist())
            current_mae = np.mean(np.abs(np.array(train_targets[-100:]) - np.array(train_preds[-100:])))
            pbar.set_postfix({"Loss": f"{loss.item() * args.accum_steps:.3f}", "Batch MAE": f"{current_mae:.2f}"})
            
        epoch_time = time.time() - epoch_start
        train_m = compute_metrics(np.array(train_targets), np.array(train_preds))
        
        # Validation Evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for (img_o, ys), (img_f, _) in tqdm(zip(val_loader_orig, val_loader_flip), total=len(val_loader_orig), desc=f"Val Ep {epoch:02d}"):
                img_o = img_o.to(device)
                img_f = img_f.to(device)
                p = 0.5 * model(img_o)["pred_age"].cpu().numpy() + 0.5 * model(img_f)["pred_age"].cpu().numpy()
                val_preds.extend(p.tolist())
                val_targets.extend(ys.numpy().tolist())
                
        val_m = compute_metrics(np.array(val_targets), np.array(val_preds))
        avg_loss = total_loss / len(train_loader)
        is_best = val_m["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_m["mae"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "val_mae": best_val_mae,
                "metrics": val_m,
                "backbone": "tf_efficientnetv2_s",
                "head": "dex"
            }, save_checkpoint_path)
            
        star = " [*] BEST (RECORD!)" if is_best else ""
        print(f"\n[+] STAGE 1 - EPOCH {epoch:02d} SUMMARY [{epoch_time:.1f}s]:", flush=True)
        print(f"    Validation MAE  : {val_m['mae']:.3f} years{star} | RMSE: {val_m['rmse']:.3f}", flush=True)
        print(f"    Accuracy @ +-3y : {val_m['acc_3']}% | Acc @ +-5y: {val_m['acc_5']}%", flush=True)
        print_age_breakdown(np.array(val_targets), np.array(val_preds), title=f"STAGE 1 - EPOCH {epoch:02d} DEMOGRAPHIC BREAKDOWN")
        
        history.append({
            "stage": 1, "epoch": epoch, "train_loss": avg_loss, "train_mae": train_m["mae"],
            "val_mae": val_m["mae"], "val_rmse": val_m["rmse"], "val_acc_3": val_m["acc_3"],
            "val_acc_5": val_m["acc_5"], "is_best": is_best
        })
        
    # =========================================================================
    # STAGE 2: Gentle Joint Fine-Tuning (5 Epochs)
    # =========================================================================
    print("=" * 85, flush=True)
    print(f" [*] STAGE 2: GENTLE JOINT FINE-TUNING ({args.stage2_epochs} Epochs | Low LR)", flush=True)
    print("=" * 85, flush=True)
    
    for param in model.backbone.parameters():
        param.requires_grad = True
        
    param_groups = model.get_parameter_groups(head_lr=8.0e-5, backbone_lr=8.0e-6, weight_decay=1e-4)
    optimizer_s2 = torch.optim.AdamW(param_groups)
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=args.stage2_epochs, eta_min=1e-6)
    
    for epoch in range(1, args.stage2_epochs + 1):
        global_ep = args.stage1_epochs + epoch
        print(f"\n{'='*30} STAGE 2 - EPOCH {epoch:02d}/{args.stage2_epochs:02d} (Global Ep {global_ep:02d}) {'='*30}", flush=True)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []
        epoch_start = time.time()
        optimizer_s2.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Stage 2 Ep {epoch:02d}", unit="batch")
        for step, (images, ages) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            ages = ages.to(device, non_blocking=True)
            target_gaussian = generate_gaussian_labels(ages, num_classes=100, sigma=1.5, device=device)
            
            with torch.cuda.amp.autocast():
                out = model(images)
                logits = out["logits"]
                pred_age = out["pred_age"]
                loss = -(target_gaussian * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                loss = loss / args.accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer_s2)
                scaler.update()
                optimizer_s2.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * args.accum_steps
            train_preds.extend(pred_age.detach().cpu().numpy().tolist())
            train_targets.extend(ages.cpu().numpy().tolist())
            current_mae = np.mean(np.abs(np.array(train_targets[-100:]) - np.array(train_preds[-100:])))
            pbar.set_postfix({"Loss": f"{loss.item() * args.accum_steps:.3f}", "Batch MAE": f"{current_mae:.2f}"})
            
        scheduler_s2.step()
        epoch_time = time.time() - epoch_start
        train_m = compute_metrics(np.array(train_targets), np.array(train_preds))
        
        # Validation Evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for (img_o, ys), (img_f, _) in tqdm(zip(val_loader_orig, val_loader_flip), total=len(val_loader_orig), desc=f"Val Ep {epoch:02d}"):
                img_o = img_o.to(device)
                img_f = img_f.to(device)
                p = 0.5 * model(img_o)["pred_age"].cpu().numpy() + 0.5 * model(img_f)["pred_age"].cpu().numpy()
                val_preds.extend(p.tolist())
                val_targets.extend(ys.numpy().tolist())
                
        val_m = compute_metrics(np.array(val_targets), np.array(val_preds))
        avg_loss = total_loss / len(train_loader)
        is_best = val_m["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_m["mae"]
            best_epoch = global_ep
            torch.save({
                "epoch": global_ep,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "val_mae": best_val_mae,
                "metrics": val_m,
                "backbone": "tf_efficientnetv2_s",
                "head": "dex"
            }, save_checkpoint_path)
            
        star = " [*] BEST (RECORD!)" if is_best else ""
        print(f"\n[+] STAGE 2 - EPOCH {epoch:02d} SUMMARY [{epoch_time:.1f}s]:", flush=True)
        print(f"    Validation MAE  : {val_m['mae']:.3f} years{star} | RMSE: {val_m['rmse']:.3f}", flush=True)
        print(f"    Accuracy @ +-3y : {val_m['acc_3']}% | Acc @ +-5y: {val_m['acc_5']}%", flush=True)
        print_age_breakdown(np.array(val_targets), np.array(val_preds), title=f"STAGE 2 - EPOCH {epoch:02d} DEMOGRAPHIC BREAKDOWN")
        
        history.append({
            "stage": 2, "epoch": global_ep, "train_loss": avg_loss, "train_mae": train_m["mae"],
            "val_mae": val_m["mae"], "val_rmse": val_m["rmse"], "val_acc_3": val_m["acc_3"],
            "val_acc_5": val_m["acc_5"], "is_best": is_best
        })
        
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    
    print("\n" + "=" * 85, flush=True)
    print(f" [RESULT] TRAINING COMPLETE! BEST OVERALL VAL MAE: {best_val_mae:.3f} YEARS (Global Epoch {best_epoch})", flush=True)
    print(f" Best Checkpoint Saved to: {save_checkpoint_path}", flush=True)
    print("=" * 85 + "\n", flush=True)

if __name__ == "__main__":
    main()
