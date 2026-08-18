"""
train_age_experts.py
Specialist Manifest Generation & Transfer Learning Fine-Tuning Pipeline
Stages 0, 1, 2, 3, 4, 5
"""
import os
import sys
import time
import math
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
from typing import Dict, Tuple, List

from models import AgeModel
from age_experts import GlobalPredictionEngine

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

def get_val_transforms(img_size: int = 320):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

class FastAgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df["image_path"].values if "image_path" in self.df.columns else self.df["filepath"].values
        self.ages = self.df["age"].values.astype(np.float32)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        age = self.ages[idx]
        
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback to black image if corrupt
            img = Image.new("RGB", (320, 320), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(age, dtype=torch.float32)

def generate_gaussian_labels(ages: torch.Tensor, num_classes: int = 100, sigma: float = 2.0, device="cuda") -> torch.Tensor:
    """
    Generates soft Gaussian target distributions for DEX cross-entropy.
    """
    bins = torch.arange(1, num_classes + 1, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 100]
    ages = ages.unsqueeze(1).to(device)  # [B, 1]
    dist_sq = (bins - ages) ** 2
    probs = torch.exp(-dist_sq / (2.0 * (sigma ** 2)))
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return probs

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = np.abs(y_true - y_pred)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    acc_3 = float(np.mean(errors <= 3.0) * 100.0)
    acc_5 = float(np.mean(errors <= 5.0) * 100.0)
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "acc_3": round(acc_3, 2),
        "acc_5": round(acc_5, 2)
    }

# =========================================================================
# STAGE 0 & 2: MANIFEST GENERATION & AUDIT
# =========================================================================
def build_specialist_manifests(master_manifest_path: str = "manifest_p2_320_plus_utkface.csv"):
    print("=" * 80)
    print("[*] STAGE 2: BUILDING SPECIALIST TRAINING MANIFESTS")
    print("=" * 80)
    
    os.makedirs("checkpoints", exist_ok=True)
    df = pd.read_csv(master_manifest_path)
    
    # Strictly training split only
    train_df = df[df["split"] == "train"].copy()
    print(f"[*] Total master training samples: {len(train_df)}")
    
    # 1. Expert 46-60
    e1_df = train_df[(train_df["age"] >= 46) & (train_df["age"] <= 60)].copy()
    e1_df["image_path"] = e1_df["filepath"]
    e1_df["source"] = e1_df["hash"].apply(lambda h: "external" if str(h).startswith("utk_") else "original")
    e1_df["age_band"] = "46-60"
    e1_path = "checkpoints/expert_46_60_train.csv"
    e1_df[["image_path", "age", "source", "age_band"]].to_csv(e1_path, index=False)
    print(f" [+] Expert 46-60 Manifest: {len(e1_df)} samples ({e1_df['source'].value_counts().to_dict()}) -> {e1_path}")
    
    # 2. Expert 61-75
    e2_df = train_df[(train_df["age"] >= 61) & (train_df["age"] <= 75)].copy()
    e2_df["image_path"] = e2_df["filepath"]
    e2_df["source"] = e2_df["hash"].apply(lambda h: "external" if str(h).startswith("utk_") else "original")
    e2_df["age_band"] = "61-75"
    e2_path = "checkpoints/expert_61_75_train.csv"
    e2_df[["image_path", "age", "source", "age_band"]].to_csv(e2_path, index=False)
    print(f" [+] Expert 61-75 Manifest: {len(e2_df)} samples ({e2_df['source'].value_counts().to_dict()}) -> {e2_path}")
    
    # 3. Expert 76-100
    e3_df = train_df[(train_df["age"] >= 76) & (train_df["age"] <= 100)].copy()
    e3_df["image_path"] = e3_df["filepath"]
    e3_df["source"] = e3_df["hash"].apply(lambda h: "external" if str(h).startswith("utk_") else "original")
    e3_df["age_band"] = "76-100"
    e3_path = "checkpoints/expert_76_100_train.csv"
    e3_df[["image_path", "age", "source", "age_band"]].to_csv(e3_path, index=False)
    print(f" [+] Expert 76-100 Manifest: {len(e3_df)} samples ({e3_df['source'].value_counts().to_dict()}) -> {e3_path}")
    
    # Verify zero val/test leakage
    val_set = set(df[df["split"] == "val"]["filepath"])
    test_set = set(df[df["split"] == "test"]["filepath"])
    for name, sp_df in [("46-60", e1_df), ("61-75", e2_df), ("76-100", e3_df)]:
        sp_set = set(sp_df["image_path"])
        val_overlap = len(sp_set.intersection(val_set))
        test_overlap = len(sp_set.intersection(test_set))
        assert val_overlap == 0, f"Error: Leak detected between {name} and validation set!"
        assert test_overlap == 0, f"Error: Leak detected between {name} and test set!"
    print("[+] Zero-leakage verification PASSED for all specialist manifests.\n")
    return e1_path, e2_path, e3_path

# =========================================================================
# STAGE 1: GLOBAL ENSEMBLE VALIDATION BASELINE
# =========================================================================
def run_global_ensemble_baseline(master_manifest_path: str = "manifest_p2_320_plus_utkface.csv", batch_size: int = 48):
    print("=" * 80)
    print("[*] STAGE 1: EVALUATING GLOBAL ENSEMBLE BASELINE ON VALIDATION SET")
    print("=" * 80)
    
    df = pd.read_csv(master_manifest_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    val_df["image_path"] = val_df["filepath"]
    print(f"[*] Total Validation Samples: {len(val_df)}")
    
    engine = GlobalPredictionEngine()
    device = engine.device
    
    ds_orig = FastAgeDataset(val_df, transform=engine.tf_orig)
    ds_flip = FastAgeDataset(val_df, transform=engine.tf_flip)
    
    loader_orig = DataLoader(ds_orig, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader_flip = DataLoader(ds_flip, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    preds_a = []
    preds_b = []
    actuals = []
    
    print("  -> Running forward pass on Model A (DEX) & Model B (Hybrid)...")
    start_t = time.time()
    
    with torch.no_grad():
        for (img_orig, y), (img_flip, _) in zip(loader_orig, loader_flip):
            img_orig = img_orig.to(device, non_blocking=True)
            img_flip = img_flip.to(device, non_blocking=True)
            
            # Model A
            out_a_o = engine.model_a(img_orig)["pred_age"].cpu().numpy()
            out_a_f = engine.model_a(img_flip)["pred_age"].cpu().numpy()
            p_a = 0.5 * out_a_o + 0.5 * out_a_f
            
            # Model B
            out_b_o = engine.model_b(img_orig)["pred_age"].cpu().numpy()
            out_b_f = engine.model_b(img_flip)["pred_age"].cpu().numpy()
            p_b = 0.5 * out_b_o + 0.5 * out_b_f
            
            preds_a.extend(p_a.tolist())
            preds_b.extend(p_b.tolist())
            actuals.extend(y.numpy().tolist())
            
    elapsed = time.time() - start_t
    print(f"[*] Inference completed in {elapsed:.1f}s ({len(val_df)/elapsed:.1f} FPS)")
    
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    actuals = np.array(actuals)
    global_pred = 0.5 * preds_a + 0.5 * preds_b
    disagreement = np.abs(preds_a - preds_b)
    abs_errors = np.abs(actuals - global_pred)
    
    m_a = compute_metrics(actuals, preds_a)
    m_b = compute_metrics(actuals, preds_b)
    m_glob = compute_metrics(actuals, global_pred)
    
    print("\n" + "-" * 70)
    print(f" * Model A (EffNetV2-S DEX) Val MAE : {m_a['mae']:.3f} yrs | RMSE: {m_a['rmse']:.3f} | Acc@±3: {m_a['acc_3']}% | Acc@±5: {m_a['acc_5']}%")
    print(f" * Model B (EffNetV2-S Hyb) Val MAE : {m_b['mae']:.3f} yrs | RMSE: {m_b['rmse']:.3f} | Acc@±3: {m_b['acc_3']}% | Acc@±5: {m_b['acc_5']}%")
    print(f" * Global Ensemble (A + B)  Val MAE : {m_glob['mae']:.3f} yrs | RMSE: {m_glob['rmse']:.3f} | Acc@±3: {m_glob['acc_3']}% | Acc@±5: {m_glob['acc_5']}%")
    print("-" * 70 + "\n")
    
    out_df = pd.DataFrame({
        "image_path": val_df["image_path"],
        "actual_age": actuals,
        "prediction_a": preds_a,
        "prediction_b": preds_b,
        "global_prediction": global_pred,
        "global_disagreement": disagreement,
        "absolute_error": abs_errors
    })
    save_path = "checkpoints/global_ensemble_validation_predictions.csv"
    out_df.to_csv(save_path, index=False)
    print(f"[+] Saved global ensemble validation predictions to: {save_path}\n")
    return m_a, m_b, m_glob

# =========================================================================
# STAGES 3, 4, 5: SPECIALIST TRANSFER LEARNING ENGINE
# =========================================================================
def train_single_specialist(
    expert_name: str,
    target_range: Tuple[int, int],
    train_manifest_path: str,
    master_manifest_path: str = "manifest_p2_320_plus_utkface.csv",
    pretrained_checkpoint: str = "outputs/exp25_effnetv2s_dex_expected_age/best_model.pt",
    epochs: int = 5,
    batch_size: int = 16,
    accum_steps: int = 2,
    lr_head: float = 3e-4,
    lr_backbone: float = 3e-5,
    device: str = "cuda"
):
    print("=" * 80)
    print(f"[*] TRAINING SPECIALIST: {expert_name} (Ages {target_range[0]} to {target_range[1]})")
    print("=" * 80)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    train_df = pd.read_csv(train_manifest_path)
    all_df = pd.read_csv(master_manifest_path)
    val_full_df = all_df[all_df["split"] == "val"].reset_index(drop=True)
    val_full_df["image_path"] = val_full_df["filepath"]
    
    # Sub-band validation set
    val_band_df = val_full_df[(val_full_df["age"] >= target_range[0]) & (val_full_df["age"] <= target_range[1])].reset_index(drop=True)
    
    print(f"[*] Specialist '{expert_name}' Data: Train={len(train_df)} | Target Val Band={len(val_band_df)} | Full Val={len(val_full_df)}")
    
    tf_train = get_train_transforms(320)
    tf_val = get_val_transforms(320)
    
    ds_train = FastAgeDataset(train_df, transform=tf_train)
    ds_val_band = FastAgeDataset(val_band_df, transform=tf_val)
    
    # Age-balanced sampler for rare age classes (especially for 76-100)
    age_counts = train_df["age"].value_counts().to_dict()
    sample_weights = [1.0 / math.sqrt(age_counts.get(a, 1.0)) for a in train_df["age"]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_band_loader = DataLoader(ds_val_band, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Instantiate Model with Transfer Learning
    print(f"[*] Initializing model with transfer weights from {pretrained_checkpoint}...")
    model = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="dex", pretrained=False).to(device)
    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Unfreeze only the last blocks and head for efficient fine-tuning
    model.unfreeze_last_n_blocks(n=3)
    param_groups = model.get_parameter_groups(head_lr=lr_head, backbone_lr=lr_backbone, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_mae = float("inf")
    best_metrics = {}
    best_epoch = 0
    save_checkpoint_path = f"checkpoints/expert_{expert_name}_best.pt"
    
    # 3. Initial Zero-Shot Band Evaluation (Before Fine-Tuning)
    model.eval()
    init_preds, init_targets = [], []
    with torch.no_grad():
        for imgs, ys in val_band_loader:
            imgs = imgs.to(device, non_blocking=True)
            p = model(imgs)["pred_age"].cpu().numpy()
            init_preds.extend(p.tolist())
            init_targets.extend(ys.numpy().tolist())
    init_m = compute_metrics(np.array(init_targets), np.array(init_preds))
    print(f"[*] Pre-Fine-Tuning Baseline on [{target_range[0]}-{target_range[1]}]: MAE = {init_m['mae']:.3f} yrs | RMSE = {init_m['rmse']:.3f}\n")
    
    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        for step, (images, ages) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            ages = ages.to(device, non_blocking=True)
            target_gaussian = generate_gaussian_labels(ages, num_classes=100, sigma=2.0, device=device)
            
            with torch.cuda.amp.autocast():
                out = model(images)
                logits = out["logits"]
                pred_age = out["pred_age"]
                loss = -(target_gaussian * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                loss = loss / accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * accum_steps
            train_preds.extend(pred_age.detach().cpu().numpy().tolist())
            train_targets.extend(ages.cpu().numpy().tolist())
            
        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_m = compute_metrics(np.array(train_targets), np.array(train_preds))
        
        # Validation on target band
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, ages in val_band_loader:
                images = images.to(device, non_blocking=True)
                out = model(images)
                val_preds.extend(out["pred_age"].cpu().numpy().tolist())
                val_targets.extend(ages.numpy().tolist())
                
        val_m = compute_metrics(np.array(val_targets), np.array(val_preds))
        avg_loss = total_loss / len(train_loader)
        
        is_best = val_m["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_m["mae"]
            best_metrics = val_m
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": best_val_mae,
                "target_range": target_range,
                "expert_name": expert_name
            }, save_checkpoint_path)
            
        star = " [*] BEST" if is_best else ""
        print(f" Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] | Loss: {avg_loss:.4f}, Train MAE: {train_m['mae']:.2f} yrs | Target Band Val MAE: {val_m['mae']:.2f} yrs, RMSE: {val_m['rmse']:.2f}, Acc@±5: {val_m['acc_5']}%{star}")
        
    print(f"\n[+] Finished Specialist '{expert_name}'. Best Epoch {best_epoch} with Target Val MAE: {best_val_mae:.3f} yrs")
    print(f"[+] Checkpoint saved to: {save_checkpoint_path}\n")
    return best_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Age Specialist Models")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per specialist")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    # Step 1: Generate Specialist Manifests (Stage 2)
    e1_path, e2_path, e3_path = build_specialist_manifests()
    
    # Step 2: Evaluate Global Ensemble Baseline (Stage 0 & 1)
    run_global_ensemble_baseline()
    
    # Step 3: Train Expert 46-60 (Stage 3)
    train_single_specialist("46_60", (46, 60), e1_path, epochs=args.epochs, batch_size=args.batch_size)
    
    # Step 4: Train Expert 61-75 (Stage 4)
    train_single_specialist("61_75", (61, 75), e2_path, epochs=args.epochs, batch_size=args.batch_size)
    
    # Step 5: Train Expert 76-100 (Stage 5)
    train_single_specialist("76_100", (76, 100), e3_path, epochs=args.epochs, batch_size=args.batch_size)
    
    print("\n" + "=" * 80)
    print("[+] ALL 3 SPECIALISTS TRAINED & SAVED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
