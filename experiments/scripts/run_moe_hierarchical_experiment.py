"""
run_moe_hierarchical_experiment.py
End-to-End Hierarchical Mixture-of-Experts (MoE) Architecture:
Champion Global Model (4.499 MAE Anchor) + 3 Age Specialists + Soft Gaussian Age Gate
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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple

from models import AgeModel
from age_experts import GlobalPredictionEngine, SpecialistModel, get_eval_transforms
from age_gate import AgeAwareGate

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

def generate_gaussian_labels(ages: torch.Tensor, num_classes: int = 100, sigma: float = 2.0, device="cuda") -> torch.Tensor:
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

def compute_age_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"target": y_true, "pred": y_pred})
    df["error"] = np.abs(df["pred"] - df["target"])
    df["acc_3"] = df["error"] <= 3.0
    df["acc_5"] = df["error"] <= 5.0
    
    bins = [0, 12, 19, 35, 60, 75, 105]
    labels = ['1-12 (Kids)', '13-19 (Teens)', '20-35 (Young Adults)', '36-60 (Middle Age)', '61-75 (Seniors)', '76-100 (Elderly)']
    df["bracket"] = pd.cut(df["target"], bins=bins, labels=labels)
    
    res = df.groupby("bracket", observed=False).agg(
        Count=("error", "count"),
        MAE=("error", "mean"),
        RMSE=("error", lambda x: np.sqrt(np.mean(x**2))),
        Acc_3=("acc_3", lambda x: np.mean(x) * 100.0),
        Acc_5=("acc_5", lambda x: np.mean(x) * 100.0)
    ).round(3)
    return res

def train_elderly_specialist(device: torch.device):
    """
    Fine-tunes Specialist 76-100 for 3 fast epochs using augmented IMDB elderly data.
    """
    ckpt_path = "checkpoints/expert_76_100_best.pt"
    manifest_path = "manifest_master_imdb_augmented.csv"
    if not os.path.exists(manifest_path):
        manifest_path = "manifest_p2_320_plus_utkface.csv"
        
    df = pd.read_csv(manifest_path)
    train_df = df[(df["split"] == "train") & (df["age"] >= 70) & (df["age"] <= 100)].reset_index(drop=True)
    val_df = df[(df["split"] == "val") & (df["age"] >= 76) & (df["age"] <= 100)].reset_index(drop=True)
    
    print("=" * 85, flush=True)
    print(f" [*] TRAINING SPECIALIST 76-100 (Elderly Focus) ON {len(train_df):,} IMAGES...", flush=True)
    print("=" * 85, flush=True)
    
    tf_train = T.Compose([
        T.Resize((350, 350)),
        T.RandomCrop((320, 320)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    tf_orig, tf_flip = get_eval_transforms(320)
    
    ds_train = FastAgeDataset(train_df, transform=tf_train)
    ds_val_orig = FastAgeDataset(val_df, transform=tf_orig)
    ds_val_flip = FastAgeDataset(val_df, transform=tf_flip)
    
    train_loader = DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader_orig = DataLoader(ds_val_orig, batch_size=32, shuffle=False, num_workers=2)
    val_loader_flip = DataLoader(ds_val_flip, batch_size=32, shuffle=False, num_workers=2)
    
    pretrained_path = "outputs/exp25_effnetv2s_dex_expected_age/best_model.pt"
    model = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="dex", pretrained=False).to(device)
    ckpt = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    param_groups = model.get_parameter_groups(head_lr=1.5e-4, backbone_lr=1.5e-5, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.cuda.amp.GradScaler()
    
    best_mae = 999.0
    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        for imgs, ages in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            ages = ages.to(device, non_blocking=True)
            target_gaussian = generate_gaussian_labels(ages, num_classes=100, sigma=2.0, device=device)
            
            with torch.cuda.amp.autocast():
                out = model(imgs)
                logits = out["logits"]
                loss = -(target_gaussian * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for (img_o, ys), (img_f, _) in zip(val_loader_orig, val_loader_flip):
                img_o = img_o.to(device)
                img_f = img_f.to(device)
                p = 0.5 * model(img_o)["pred_age"].cpu().numpy() + 0.5 * model(img_f)["pred_age"].cpu().numpy()
                val_preds.extend(p.tolist())
                val_targets.extend(ys.numpy().tolist())
                
        val_mae = float(np.mean(np.abs(np.array(val_targets) - np.array(val_preds))))
        print(f" [+] Specialist 76-100 Epoch {epoch:02d}/03 | Target Band Val MAE: {val_mae:.3f} yrs", flush=True)
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"model_state_dict": model.state_dict(), "val_mae": val_mae}, ckpt_path)
            
    print(f"[+] Specialist 76-100 Saved to {ckpt_path} (Best Target MAE: {best_mae:.3f} yrs)\n", flush=True)

def evaluate_moe_pipeline(
    manifest_path: str = "manifest_p2_320_plus_utkface.csv",
    split: str = "val",
    batch_size: int = 48,
    device: str = "cuda"
) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
    """
    Evaluates Champion Global Model (4.499 Anchor) and 3 Specialists with 2-View TTA.
    """
    cache_path = f"checkpoints/eval_moe_{split}_cache.npz"
    df_all = pd.read_csv(manifest_path)
    df_split = df_all[df_all["split"] == split].reset_index(drop=True)
    df_split["image_path"] = df_split["filepath"]
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating {len(df_split):,} images in '{split}' split on {device}...", flush=True)
    
    tf_orig, tf_flip = get_eval_transforms(320)
    ds_orig = FastAgeDataset(df_split, transform=tf_orig)
    ds_flip = FastAgeDataset(df_split, transform=tf_flip)
    
    loader_orig = DataLoader(ds_orig, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader_flip = DataLoader(ds_flip, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 1. Global Champion Model (Anchor)
    global_model = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="dex", pretrained=False).to(device)
    ckpt_glob = torch.load("outputs/exp25_effnetv2s_dex_expected_age/best_model.pt", map_location=device)
    global_model.load_state_dict(ckpt_glob["model_state_dict"])
    global_model.eval()
    
    # 2. Specialist Models
    exp_46_60 = SpecialistModel(target_range=(46, 60), pretrained_path="checkpoints/expert_46_60_best.pt").to(device)
    exp_46_60.eval()
    
    exp_61_75 = SpecialistModel(target_range=(61, 75), pretrained_path="checkpoints/expert_61_75_best.pt").to(device)
    exp_61_75.eval()
    
    exp_76_100 = SpecialistModel(target_range=(76, 100), pretrained_path="checkpoints/expert_76_100_best.pt").to(device)
    exp_76_100.eval()
    
    preds = {
        "global_pred": [],
        "expert_46_60": [],
        "expert_61_75": [],
        "expert_76_100": []
    }
    targets = []
    
    start_t = time.time()
    with torch.no_grad():
        for (img_o, ys), (img_f, _) in tqdm(zip(loader_orig, loader_flip), total=len(loader_orig), desc=f"Evaluating '{split}'"):
            img_o = img_o.to(device, non_blocking=True)
            img_f = img_f.to(device, non_blocking=True)
            
            # Global Anchor
            out_g_o = global_model(img_o)["pred_age"].cpu().numpy()
            out_g_f = global_model(img_f)["pred_age"].cpu().numpy()
            preds["global_pred"].extend((0.5 * out_g_o + 0.5 * out_g_f).tolist())
            
            # Expert 46-60
            out_e1_o = exp_46_60(img_o)["pred_age"].cpu().numpy()
            out_e1_f = exp_46_60(img_f)["pred_age"].cpu().numpy()
            preds["expert_46_60"].extend((0.5 * out_e1_o + 0.5 * out_e1_f).tolist())
            
            # Expert 61-75
            out_e2_o = exp_61_75(img_o)["pred_age"].cpu().numpy()
            out_e2_f = exp_61_75(img_f)["pred_age"].cpu().numpy()
            preds["expert_61_75"].extend((0.5 * out_e2_o + 0.5 * out_e2_f).tolist())
            
            # Expert 76-100
            out_e3_o = exp_76_100(img_o)["pred_age"].cpu().numpy()
            out_e3_f = exp_76_100(img_f)["pred_age"].cpu().numpy()
            preds["expert_76_100"].extend((0.5 * out_e3_o + 0.5 * out_e3_f).tolist())
            
            targets.extend(ys.numpy().tolist())
            
    elapsed = time.time() - start_t
    print(f"[*] Inference on {len(df_split):,} samples completed in {elapsed:.1f}s ({len(df_split)/elapsed:.1f} FPS)!\n", flush=True)
    
    targets = np.array(targets)
    for k in preds:
        preds[k] = np.array(preds[k])
        
    return targets, preds, df_split

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Mixture-of-Experts Experiment")
    parser.add_argument("--skip_train_elderly", action="store_true", help="Skip 3-epoch elderly fine-tuning")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90, flush=True)
    print(" 🚀 HIERARCHICAL MIXTURE-OF-EXPERTS (MoE) EXPERIMENT", flush=True)
    print(" Global Anchor Model (4.499 MAE) + 3 Specialists (46-60, 61-75, 76-100) + Soft Age Gate", flush=True)
    print("=" * 90, flush=True)
    
    # Step 1: Polish Elderly Specialist if needed
    if not args.skip_train_elderly:
        train_elderly_specialist(device)
        
    # Step 2: Full Validation Inference (16,258 Benchmark Faces)
    val_targets, val_preds, val_df = evaluate_moe_pipeline(split="val", device=str(device))
    
    # Step 3: Compute Baseline Global Anchor Performance
    m_glob = compute_metrics(val_targets, val_preds["global_pred"])
    print("=" * 90, flush=True)
    print(f" [*] CHAMPION GLOBAL MODEL (ANCHOR) VALIDATION SCORE:", flush=True)
    print(f"     MAE = {m_glob['mae']:.3f} years | RMSE = {m_glob['rmse']:.3f} | Acc@+-3 = {m_glob['acc_3']}% | Acc@+-5 = {m_glob['acc_5']}%", flush=True)
    print("=" * 90 + "\n", flush=True)
    
    # Step 4: Soft Age Gating Optimization (Grid Search on Validation Data)
    print("=" * 90, flush=True)
    print(" [*] OPTIMIZING SOFT AGE-AWARE GATE WEIGHTS...", flush=True)
    print("=" * 90, flush=True)
    
    best_moe_mae = 999.0
    best_config = None
    best_fused_val = None
    
    # Grid search across global anchor minimum weights and Gaussian widths (sigma)
    for gw_min in [0.40, 0.50, 0.60, 0.70, 0.80]:
        for sig in [5.0, 7.0, 8.0, 10.0, 12.0]:
            gate = AgeAwareGate(center_46_60=53.0, center_61_75=68.0, center_76_100=88.0, sigma=sig, global_weight_min=gw_min)
            
            fused = []
            for gp, e1, e2, e3 in zip(val_preds["global_pred"], val_preds["expert_46_60"], val_preds["expert_61_75"], val_preds["expert_76_100"]):
                w = gate.compute_weights(global_age=gp)
                f = w["global"] * gp + w["46_60"] * e1 + w["61_75"] * e2 + w["76_100"] * e3
                fused.append(f)
                
            fused = np.array(fused)
            m_fused = compute_metrics(val_targets, fused)
            
            if m_fused["mae"] < best_moe_mae:
                best_moe_mae = m_fused["mae"]
                best_config = {"gw_min": gw_min, "sigma": sig, "metrics": m_fused}
                best_fused_val = fused
                
    print(f"[+] BEST SOFT GATE CONFIGURATION: Global Anchor Min Weight = {best_config['gw_min']:.2f} | Sigma = {best_config['sigma']:.1f}", flush=True)
    print(f"    Hierarchical MoE Val MAE  : {best_config['metrics']['mae']:.3f} years", flush=True)
    print(f"    Hierarchical MoE Val RMSE : {best_config['metrics']['rmse']:.3f}", flush=True)
    print(f"    Hierarchical MoE Acc@+-3y : {best_config['metrics']['acc_3']}%", flush=True)
    print(f"    Hierarchical MoE Acc@+-5y : {best_config['metrics']['acc_5']}%\n", flush=True)
    
    # Step 5: Side-by-Side Demographic Age Breakdown Table
    print("=" * 90, flush=True)
    print("   🏆 SIDE-BY-SIDE DEMOGRAPHIC BREAKDOWN: GLOBAL ANCHOR VS. HIERARCHICAL MoE", flush=True)
    print("=" * 90, flush=True)
    
    bd_glob = compute_age_breakdown(val_targets, val_preds["global_pred"])
    bd_moe = compute_age_breakdown(val_targets, best_fused_val)
    
    comp_df = pd.DataFrame({
        "Count": bd_glob["Count"],
        "Global MAE": bd_glob["MAE"],
        "MoE MAE": bd_moe["MAE"],
        "Diff MAE": (bd_moe["MAE"] - bd_glob["MAE"]).round(3),
        "Global Acc@+-5": bd_glob["Acc_5"],
        "MoE Acc@+-5": bd_moe["Acc_5"],
        "Diff Acc@+-5": (bd_moe["Acc_5"] - bd_glob["Acc_5"]).round(2)
    })
    print(comp_df.to_string(), flush=True)
    print("=" * 90 + "\n", flush=True)
    
    # Step 6: Final Test Set Evaluation (47,568 Untouched Locked Images)
    print("=" * 90, flush=True)
    print(" [*] FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET (47,568 IMAGES)...", flush=True)
    print("=" * 90, flush=True)
    
    test_targets, test_preds, test_df = evaluate_moe_pipeline(split="test", device=str(device))
    
    gate_final = AgeAwareGate(
        center_46_60=53.0, center_61_75=68.0, center_76_100=88.0,
        sigma=best_config["sigma"], global_weight_min=best_config["gw_min"]
    )
    
    test_moe_fused = []
    for gp, e1, e2, e3 in zip(test_preds["global_pred"], test_preds["expert_46_60"], test_preds["expert_61_75"], test_preds["expert_76_100"]):
        w = gate_final.compute_weights(global_age=gp)
        f = w["global"] * gp + w["46_60"] * e1 + w["61_75"] * e2 + w["76_100"] * e3
        test_moe_fused.append(f)
    test_moe_fused = np.array(test_moe_fused)
    
    m_test_glob = compute_metrics(test_targets, test_preds["global_pred"])
    m_test_moe = compute_metrics(test_targets, test_moe_fused)
    
    print("\n" + "=" * 90, flush=True)
    print("                    FINAL UNTOUCHED TEST SET PERFORMANCE", flush=True)
    print("=" * 90, flush=True)
    print(f" * Champion Global Model Test MAE : {m_test_glob['mae']:.3f} yrs | RMSE: {m_test_glob['rmse']:.3f} | Acc@+-5: {m_test_glob['acc_5']}%", flush=True)
    print(f" * Hierarchical MoE Test MAE      : {m_test_moe['mae']:.3f} yrs | RMSE: {m_test_moe['rmse']:.3f} | Acc@+-5: {m_test_moe['acc_5']}%", flush=True)
    print("=" * 90 + "\n", flush=True)
    
    # Save Master Comparison
    comp_records = [
        {"Model": "Champion Global Model (Anchor)", "Validation MAE": m_glob["mae"], "Validation RMSE": m_glob["rmse"], "Val Acc@+-5": m_glob["acc_5"], "Test MAE": m_test_glob["mae"], "Test RMSE": m_test_glob["rmse"], "Test Acc@+-5": m_test_glob["acc_5"]},
        {"Model": "Hierarchical MoE (Global + 3 Specialists + Soft Gate)", "Validation MAE": best_config["metrics"]["mae"], "Validation RMSE": best_config["metrics"]["rmse"], "Val Acc@+-5": best_config["metrics"]["acc_5"], "Test MAE": m_test_moe["mae"], "Test RMSE": m_test_moe["rmse"], "Test Acc@+-5": m_test_moe["acc_5"]},
    ]
    df_final = pd.DataFrame(comp_records)
    out_csv = "checkpoints/final_moe_master_comparison.csv"
    df_final.to_csv(out_csv, index=False)
    print(f"[+] Master comparison saved to: {out_csv}", flush=True)

if __name__ == "__main__":
    main()
