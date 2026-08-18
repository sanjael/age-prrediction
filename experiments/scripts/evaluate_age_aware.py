"""
evaluate_age_aware.py
Validation Gating Optimization, Age-Band Analysis & Final Untouched Test Set Evaluation
Stages 8, 9, 10, 11
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from age_experts import GlobalPredictionEngine, SpecialistModel, get_eval_transforms
from age_gate import AgeAwareGate
from train_age_experts import FastAgeDataset, compute_metrics

def evaluate_all_models_on_split(
    manifest_path: str = "manifest_p2_320_plus_utkface.csv",
    split: str = "val",
    batch_size: int = 48,
    device: str = "cuda"
) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
    """
    Runs inference across Global Model A, Model B, Expert 46-60, Expert 61-75, Expert 76-100.
    Caches predictions to disk for speed.
    """
    cache_path = f"checkpoints/eval_{split}_cache.npz"
    df_all = pd.read_csv(manifest_path)
    df_split = df_all[df_all["split"] == split].reset_index(drop=True)
    df_split["image_path"] = df_split["filepath"]
    
    if os.path.exists(cache_path):
        print(f"[*] Loading cached '{split}' predictions from {cache_path}...")
        data = np.load(cache_path)
        targets = data["targets"]
        preds = {
            "pred_a": data["pred_a"],
            "pred_b": data["pred_b"],
            "expert_46_60": data["expert_46_60"],
            "expert_61_75": data["expert_61_75"],
            "expert_76_100": data["expert_76_100"],
            "global_pred": data["global_pred"],
            "disagreement": data["disagreement"]
        }
        return targets, preds, df_split

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating {len(df_split)} images in '{split}' split on {device}...")
    
    tf_orig, tf_flip = get_eval_transforms(320)
    ds_orig = FastAgeDataset(df_split, transform=tf_orig)
    ds_flip = FastAgeDataset(df_split, transform=tf_flip)
    
    loader_orig = DataLoader(ds_orig, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader_flip = DataLoader(ds_flip, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 1. Global Engine
    global_engine = GlobalPredictionEngine(device=device)
    
    # 2. Specialist Models
    exp_46_60 = SpecialistModel(target_range=(46, 60), pretrained_path="checkpoints/expert_46_60_best.pt").to(device)
    exp_46_60.eval()
    
    exp_61_75 = SpecialistModel(target_range=(61, 75), pretrained_path="checkpoints/expert_61_75_best.pt").to(device)
    exp_61_75.eval()
    
    exp_76_100 = SpecialistModel(target_range=(76, 100), pretrained_path="checkpoints/expert_76_100_best.pt").to(device)
    exp_76_100.eval()
    
    preds = {
        "pred_a": [],
        "pred_b": [],
        "expert_46_60": [],
        "expert_61_75": [],
        "expert_76_100": []
    }
    targets = []
    
    start_t = time.time()
    with torch.no_grad():
        for (img_o, ys), (img_f, _) in zip(loader_orig, loader_flip):
            img_o = img_o.to(device, non_blocking=True)
            img_f = img_f.to(device, non_blocking=True)
            
            # Model A
            out_a_o = global_engine.model_a(img_o)["pred_age"].cpu().numpy()
            out_a_f = global_engine.model_a(img_f)["pred_age"].cpu().numpy()
            preds["pred_a"].extend((0.5 * out_a_o + 0.5 * out_a_f).tolist())
            
            # Model B
            out_b_o = global_engine.model_b(img_o)["pred_age"].cpu().numpy()
            out_b_f = global_engine.model_b(img_f)["pred_age"].cpu().numpy()
            preds["pred_b"].extend((0.5 * out_b_o + 0.5 * out_b_f).tolist())
            
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
    print(f"[*] Inference on {len(df_split)} samples completed in {elapsed:.1f}s ({len(df_split)/elapsed:.1f} FPS)!\n")
    
    targets = np.array(targets)
    for k in preds:
        preds[k] = np.array(preds[k])
        
    preds["global_pred"] = 0.5 * preds["pred_a"] + 0.5 * preds["pred_b"]
    preds["disagreement"] = np.abs(preds["pred_a"] - preds["pred_b"])
    
    # Save cache
    np.savez_compressed(
        cache_path,
        targets=targets,
        pred_a=preds["pred_a"],
        pred_b=preds["pred_b"],
        expert_46_60=preds["expert_46_60"],
        expert_61_75=preds["expert_61_75"],
        expert_76_100=preds["expert_76_100"],
        global_pred=preds["global_pred"],
        disagreement=preds["disagreement"]
    )
    print(f"[+] Cached predictions to {cache_path}")
    
    return targets, preds, df_split

def compute_age_band_metrics(targets: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"target": targets, "pred": predictions})
    df["error"] = np.abs(df["pred"] - df["target"])
    df["acc_3"] = df["error"] <= 3.0
    df["acc_5"] = df["error"] <= 5.0
    
    bins = [0, 12, 19, 30, 45, 60, 75, 105]
    labels = ['1-12 (Children)', '13-19 (Teens)', '20-30 (Young Adults)', '31-45 (Adults)', '46-60 (Middle Age)', '61-75 (Seniors)', '76-100 (Elderly)']
    df["age_group"] = pd.cut(df["target"], bins=bins, labels=labels)
    
    res = df.groupby("age_group", observed=False).agg(
        count=("error", "count"),
        mae=("error", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
        acc_3=("acc_3", lambda x: np.mean(x) * 100.0),
        acc_5=("acc_5", lambda x: np.mean(x) * 100.0)
    ).round(3)
    return res

def run_validation_optimization(targets: np.ndarray, preds: Dict[str, np.ndarray]):
    print("=" * 80)
    print("[*] STAGE 8: VALIDATION GATING OPTIMIZATION & GRID SEARCH")
    print("=" * 80)
    
    results = []
    
    # 1. Config A: Global Only
    m_glob = compute_metrics(targets, preds["global_pred"])
    results.append({
        "configuration": "Config A (Global Ensemble Only)",
        "global_weight": 1.0,
        "sigma": 0.0,
        "disagreement_weight": 0.0,
        "validation_mae": m_glob["mae"],
        "validation_rmse": m_glob["rmse"],
        "within_3": m_glob["acc_3"],
        "within_5": m_glob["acc_5"]
    })
    
    # 2. Config B: Uniform Blend (25% each)
    uniform_pred = 0.25 * preds["global_pred"] + 0.25 * preds["expert_46_60"] + 0.25 * preds["expert_61_75"] + 0.25 * preds["expert_76_100"]
    m_uni = compute_metrics(targets, uniform_pred)
    results.append({
        "configuration": "Config B (Uniform 25% Blend)",
        "global_weight": 0.25,
        "sigma": 0.0,
        "disagreement_weight": 0.0,
        "validation_mae": m_uni["mae"],
        "validation_rmse": m_uni["rmse"],
        "within_3": m_uni["acc_3"],
        "within_5": m_uni["acc_5"]
    })
    
    # 3. Config C: Soft Gaussian Age Gating Grid Search
    global_weights = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    sigmas = [5.0, 8.0, 10.0, 12.0, 15.0]
    
    for gw in global_weights:
        for sig in sigmas:
            gate = AgeAwareGate(sigma=sig, global_weight_min=gw, disagreement_scale=0.0)
            
            fused = []
            for gp, e1, e2, e3 in zip(preds["global_pred"], preds["expert_46_60"], preds["expert_61_75"], preds["expert_76_100"]):
                w = gate.compute_weights(global_age=gp, global_disagreement=0.0)
                f = w["global"] * gp + w["46_60"] * e1 + w["61_75"] * e2 + w["76_100"] * e3
                fused.append(f)
                
            fused = np.array(fused)
            m_fused = compute_metrics(targets, fused)
            results.append({
                "configuration": f"Config C (Gaussian Gating gw={gw:.2f} sig={sig:.1f})",
                "global_weight": gw,
                "sigma": sig,
                "disagreement_weight": 0.0,
                "validation_mae": m_fused["mae"],
                "validation_rmse": m_fused["rmse"],
                "within_3": m_fused["acc_3"],
                "within_5": m_fused["acc_5"]
            })
            
    # 4. Config D: Gating + Disagreement Scaling
    for dis_scale in [0.20, 0.50, 1.0]:
        best_c_gw = 0.50
        best_c_sig = 8.0
        gate = AgeAwareGate(sigma=best_c_sig, global_weight_min=best_c_gw, disagreement_scale=dis_scale)
        
        fused = []
        for gp, dis, e1, e2, e3 in zip(preds["global_pred"], preds["disagreement"], preds["expert_46_60"], preds["expert_61_75"], preds["expert_76_100"]):
            w = gate.compute_weights(global_age=gp, global_disagreement=dis)
            f = w["global"] * gp + w["46_60"] * e1 + w["61_75"] * e2 + w["76_100"] * e3
            fused.append(f)
            
        fused = np.array(fused)
        m_fused = compute_metrics(targets, fused)
        results.append({
            "configuration": f"Config D (Gating + Disagreement scale={dis_scale:.2f})",
            "global_weight": best_c_gw,
            "sigma": best_c_sig,
            "disagreement_weight": dis_scale,
            "validation_mae": m_fused["mae"],
            "validation_rmse": m_fused["rmse"],
            "within_3": m_fused["acc_3"],
            "within_5": m_fused["acc_5"]
        })
        
    df_res = pd.DataFrame(results).sort_values("validation_mae", ascending=True).reset_index(drop=True)
    save_path = "checkpoints/gating_experiment_results.csv"
    df_res.to_csv(save_path, index=False)
    print(f"[+] Saved grid search results to: {save_path}\n")
    
    print("--- TOP 5 GATING CONFIGURATIONS BY VALIDATION MAE ---")
    print(df_res.head(5).to_string())
    print("-" * 70 + "\n")
    
    best_row = df_res.iloc[0]
    print(f"[CHAMPION] BEST VALIDATION CONFIGURATION: {best_row['configuration']}")
    print(f"   Validation MAE = {best_row['validation_mae']:.3f} yrs | RMSE = {best_row['validation_rmse']:.3f} | Acc@+-5 = {best_row['within_5']}%\n")
    return best_row, df_res

def main():
    print("=" * 80)
    print("      AGE-AWARE HIERARCHICAL ENSEMBLE: FULL EVALUATION PIPELINE")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # 1. EVALUATE ON VALIDATION SPLIT (16,258 Images)
    # -------------------------------------------------------------
    val_targets, val_preds, val_df = evaluate_all_models_on_split(split="val")
    
    # Run Grid Search Optimization on Validation Data Only
    best_config, df_grid = run_validation_optimization(val_targets, val_preds)
    
    # Identify the Best Gating (Non-Global-Only) configuration to compare
    gated_configs = df_grid[df_grid["configuration"] != "Config A (Global Ensemble Only)"]
    best_gated_row = gated_configs.iloc[0] if len(gated_configs) > 0 else best_config
    
    best_gw = float(best_gated_row["global_weight"])
    best_sig = float(best_gated_row["sigma"])
    best_dis = float(best_gated_row["disagreement_weight"])
    
    # Apply Best Gate on Validation
    best_gate = AgeAwareGate(sigma=best_sig, global_weight_min=best_gw, disagreement_scale=best_dis)
    val_age_aware_pred = []
    for gp, dis, e1, e2, e3 in zip(val_preds["global_pred"], val_preds["disagreement"], val_preds["expert_46_60"], val_preds["expert_61_75"], val_preds["expert_76_100"]):
        w = best_gate.compute_weights(global_age=gp, global_disagreement=dis)
        f = w["global"] * gp + w["46_60"] * e1 + w["61_75"] * e2 + w["76_100"] * e3
        val_age_aware_pred.append(f)
    val_age_aware_pred = np.array(val_age_aware_pred)
    
    # -------------------------------------------------------------
    # 2. STAGE 9: AGE-BAND DEMOGRAPHIC COMPARISON (VALIDATION)
    # -------------------------------------------------------------
    print("=" * 80)
    print("   STAGE 9: DEMOGRAPHIC AGE-BAND COMPARISON (VALIDATION SPLIT)")
    print("=" * 80)
    
    band_global_val = compute_age_band_metrics(val_targets, val_preds["global_pred"])
    band_age_aware_val = compute_age_band_metrics(val_targets, val_age_aware_pred)
    
    comp_val = pd.DataFrame({
        "Count": band_global_val["count"],
        "Global MAE": band_global_val["mae"],
        "Age-Aware MAE": band_age_aware_val["mae"],
        "Diff MAE": (band_age_aware_val["mae"] - band_global_val["mae"]).round(3),
        "Global Acc@+-5": band_global_val["acc_5"],
        "Age-Aware Acc@+-5": band_age_aware_val["acc_5"]
    })
    print(comp_val.to_string())
    print("=" * 80 + "\n")
    
    # -------------------------------------------------------------
    # 3. STAGE 10: MODEL SELECTION DECISION
    # -------------------------------------------------------------
    val_mae_glob = compute_metrics(val_targets, val_preds["global_pred"])["mae"]
    val_mae_moe = compute_metrics(val_targets, val_age_aware_pred)["mae"]
    
    print("=" * 80)
    print(f"[*] STAGE 10: FINAL MODEL SELECTION DECISION")
    print(f"    Global Ensemble Val MAE    : {val_mae_glob:.3f} years")
    print(f"    Age-Aware Ensemble Val MAE : {val_mae_moe:.3f} years")
    
    if val_mae_moe < val_mae_glob:
        print(f"[VERDICT] Age-Aware Ensemble WINS by {val_mae_glob - val_mae_moe:.3f} years! Recommended for deployment.")
    else:
        print(f"[VERDICT] Global Ensemble retains superior/comparable performance ({val_mae_glob:.3f} vs {val_mae_moe:.3f}). Production model: Global Ensemble.")
    print("=" * 80 + "\n")
    
    # -------------------------------------------------------------
    # 4. STAGE 11: FINAL EVALUATION ON UNTOUCHED TEST SET (47,568 Images)
    # -------------------------------------------------------------
    print("=" * 80)
    print("[*] STAGE 11: FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET (47,568 IMAGES)")
    print("=" * 80)
    
    test_targets, test_preds, test_df = evaluate_all_models_on_split(split="test")
    
    # Evaluate Global Ensemble on Test
    test_global_pred = test_preds["global_pred"]
    m_test_glob = compute_metrics(test_targets, test_global_pred)
    
    # Save Global Test Predictions
    df_test_glob = pd.DataFrame({
        "image_path": test_df["image_path"],
        "actual_age": test_targets,
        "predicted_age": test_global_pred,
        "error": np.abs(test_targets - test_global_pred)
    })
    df_test_glob.to_csv("checkpoints/final_test_global.csv", index=False)
    print("[+] Saved Global Test predictions to: checkpoints/final_test_global.csv")
    
    # Apply Frozen Best Gate on Test
    test_age_aware_pred = []
    for gp, dis, e1, e2, e3 in zip(test_preds["global_pred"], test_preds["disagreement"], test_preds["expert_46_60"], test_preds["expert_61_75"], test_preds["expert_76_100"]):
        w = best_gate.compute_weights(global_age=gp, global_disagreement=dis)
        f = w["global"] * gp + w["46_60"] * e1 + w["61_75"] * e2 + w["76_100"] * e3
        test_age_aware_pred.append(f)
    test_age_aware_pred = np.array(test_age_aware_pred)
    m_test_moe = compute_metrics(test_targets, test_age_aware_pred)
    
    # Save Age-Aware Test Predictions
    df_test_moe = pd.DataFrame({
        "image_path": test_df["image_path"],
        "actual_age": test_targets,
        "predicted_age": test_age_aware_pred,
        "error": np.abs(test_targets - test_age_aware_pred)
    })
    df_test_moe.to_csv("checkpoints/final_test_age_aware.csv", index=False)
    print("[+] Saved Age-Aware Test predictions to: checkpoints/final_test_age_aware.csv\n")
    
    # Compute Individual Models on Test
    m_test_a = compute_metrics(test_targets, test_preds["pred_a"])
    m_test_b = compute_metrics(test_targets, test_preds["pred_b"])
    m_test_e1 = compute_metrics(test_targets, test_preds["expert_46_60"])
    m_test_e2 = compute_metrics(test_targets, test_preds["expert_61_75"])
    m_test_e3 = compute_metrics(test_targets, test_preds["expert_76_100"])
    
    # Individual Validation Metrics
    m_val_a = compute_metrics(val_targets, val_preds["pred_a"])
    m_val_b = compute_metrics(val_targets, val_preds["pred_b"])
    m_val_glob = compute_metrics(val_targets, val_preds["global_pred"])
    m_val_e1 = compute_metrics(val_targets, val_preds["expert_46_60"])
    m_val_e2 = compute_metrics(val_targets, val_preds["expert_61_75"])
    m_val_e3 = compute_metrics(val_targets, val_preds["expert_76_100"])
    m_val_moe = compute_metrics(val_targets, val_age_aware_pred)
    
    # Age-Band Metrics on Test Set
    def get_band_mae(tgts, prds, low, high):
        mask = (tgts >= low) & (tgts <= high)
        return round(float(np.mean(np.abs(tgts[mask] - prds[mask]))), 3) if np.sum(mask) > 0 else 0.0
        
    # Build Master Final Comparison Table
    models_summary = [
        ("Model A (EffNetV2-S DEX)", m_val_a["mae"], m_test_a["mae"], m_test_a["rmse"], m_test_a["acc_3"], m_test_a["acc_5"],
         get_band_mae(test_targets, test_preds["pred_a"], 1, 45),
         get_band_mae(test_targets, test_preds["pred_a"], 46, 60),
         get_band_mae(test_targets, test_preds["pred_a"], 61, 75),
         get_band_mae(test_targets, test_preds["pred_a"], 76, 100)),
        ("Model B (EffNetV2-S Hybrid)", m_val_b["mae"], m_test_b["mae"], m_test_b["rmse"], m_test_b["acc_3"], m_test_b["acc_5"],
         get_band_mae(test_targets, test_preds["pred_b"], 1, 45),
         get_band_mae(test_targets, test_preds["pred_b"], 46, 60),
         get_band_mae(test_targets, test_preds["pred_b"], 61, 75),
         get_band_mae(test_targets, test_preds["pred_b"], 76, 100)),
        ("Global Ensemble (A + B)", m_val_glob["mae"], m_test_glob["mae"], m_test_glob["rmse"], m_test_glob["acc_3"], m_test_glob["acc_5"],
         get_band_mae(test_targets, test_global_pred, 1, 45),
         get_band_mae(test_targets, test_global_pred, 46, 60),
         get_band_mae(test_targets, test_global_pred, 61, 75),
         get_band_mae(test_targets, test_global_pred, 76, 100)),
        ("Expert 46-60", m_val_e1["mae"], m_test_e1["mae"], m_test_e1["rmse"], m_test_e1["acc_3"], m_test_e1["acc_5"],
         get_band_mae(test_targets, test_preds["expert_46_60"], 1, 45),
         get_band_mae(test_targets, test_preds["expert_46_60"], 46, 60),
         get_band_mae(test_targets, test_preds["expert_46_60"], 61, 75),
         get_band_mae(test_targets, test_preds["expert_46_60"], 76, 100)),
        ("Expert 61-75", m_val_e2["mae"], m_test_e2["mae"], m_test_e2["rmse"], m_test_e2["acc_3"], m_test_e2["acc_5"],
         get_band_mae(test_targets, test_preds["expert_61_75"], 1, 45),
         get_band_mae(test_targets, test_preds["expert_61_75"], 46, 60),
         get_band_mae(test_targets, test_preds["expert_61_75"], 61, 75),
         get_band_mae(test_targets, test_preds["expert_61_75"], 76, 100)),
        ("Expert 76-100", m_val_e3["mae"], m_test_e3["mae"], m_test_e3["rmse"], m_test_e3["acc_3"], m_test_e3["acc_5"],
         get_band_mae(test_targets, test_preds["expert_76_100"], 1, 45),
         get_band_mae(test_targets, test_preds["expert_76_100"], 46, 60),
         get_band_mae(test_targets, test_preds["expert_76_100"], 61, 75),
         get_band_mae(test_targets, test_preds["expert_76_100"], 76, 100)),
        ("Age-Aware Ensemble", m_val_moe["mae"], m_test_moe["mae"], m_test_moe["rmse"], m_test_moe["acc_3"], m_test_moe["acc_5"],
         get_band_mae(test_targets, test_age_aware_pred, 1, 45),
         get_band_mae(test_targets, test_age_aware_pred, 46, 60),
         get_band_mae(test_targets, test_age_aware_pred, 61, 75),
         get_band_mae(test_targets, test_age_aware_pred, 76, 100)),
    ]
    
    df_comparison = pd.DataFrame(models_summary, columns=[
        "Model", "Validation MAE", "Test MAE", "Test RMSE", "Within +-3", "Within +-5",
        "1-45 MAE", "46-60 MAE", "61-75 MAE", "76-100 MAE"
    ])
    
    comp_save_path = "checkpoints/final_experiment_comparison.csv"
    df_comparison.to_csv(comp_save_path, index=False)
    print(f"[+] Saved final master comparison to: {comp_save_path}\n")
    
    print("=" * 110)
    print("                          FINAL EXPERIMENT COMPARISON MASTER TABLE")
    print("=" * 110)
    print(df_comparison.to_string(index=False))
    print("=" * 110 + "\n")

if __name__ == "__main__":
    main()
