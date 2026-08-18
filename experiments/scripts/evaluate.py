import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import Config
from utils import (
    compute_metrics,
    compute_age_bin_metrics,
    plot_predicted_vs_true,
    plot_error_distribution,
    plot_per_age_mae
)
from dataset import get_dataloaders
from models import AgeModel

def evaluate_checkpoint(
    checkpoint_path: str,
    manifest_path: str = None,
    split: str = "test",
    output_dir: str = None,
    batch_size: int = 32
):
    cfg = Config()
    if manifest_path:
        cfg.manifest_path = manifest_path
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_cfg = checkpoint.get("config", {})
    
    backbone = saved_cfg.get("backbone", cfg.backbone)
    head_type = saved_cfg.get("head_type", cfg.head_type)
    num_classes = saved_cfg.get("num_classes", cfg.num_classes)
    
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Model
    model = AgeModel(
        backbone_name=backbone,
        head_type=head_type,
        pretrained=False,
        num_classes=num_classes
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 2. Load Data
    dataloaders = get_dataloaders(cfg)
    target_loader = dataloaders[split]
    target_df = dataloaders[f"{split}_df"].reset_index(drop=True)
    
    print(f"\nEvaluating on '{split}' split ({len(target_df):,} samples)...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, age_targets, _ in tqdm(target_loader, desc=f"Evaluating {split}"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs["pred_age"].detach().cpu().numpy()
            targets = age_targets.numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 3. Overall Metrics
    metrics = compute_metrics(all_preds, all_targets)
    
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY ({split.upper()} SET)")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE):    {metrics['mae']:.2f} years")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f} years")
    print(f"Coefficient of Determ. (R²):   {metrics['r2']:.4f}")
    print(f"Accuracy within +/-1 year:       {metrics['acc_1']:.2f}%")
    print(f"Accuracy within +/-3 years:      {metrics['acc_3']:.2f}%")
    print(f"Accuracy within +/-5 years:      {metrics['acc_5']:.2f}%")
    print(f"Accuracy within +/-10 years:     {metrics['acc_10']:.2f}%")
    print("=" * 60)
    
    # 4. Per-Age-Bin Breakdown
    bin_table = compute_age_bin_metrics(all_preds, all_targets)
    print("\nPer-Age-Group Performance:")
    print(bin_table.to_string(index=False))
    
    # 5. Top 15 Worst Predictions
    eval_df = target_df.copy()
    eval_df["predicted_age"] = np.round(all_preds, 2)
    eval_df["abs_error"] = np.round(np.abs(all_preds - all_targets), 2)
    eval_df["error_direction"] = np.where(all_preds > all_targets, "Overestimated", "Underestimated")
    
    worst_df = eval_df.sort_values(by="abs_error", ascending=False).head(15)
    print("\nTop 15 Largest Absolute Errors:")
    print(worst_df[["filename", "age", "predicted_age", "abs_error", "error_direction"]].to_string(index=False))
    
    # 6. Save Artifacts
    # Save predictions CSV
    preds_csv_path = os.path.join(output_dir, f"{split}_predictions.csv")
    eval_df.to_csv(preds_csv_path, index=False)
    print(f"\nSaved full predictions to: {preds_csv_path}")
    
    # Save worst errors CSV
    worst_csv_path = os.path.join(output_dir, f"{split}_worst_errors.csv")
    worst_df.to_csv(worst_csv_path, index=False)
    
    # Save text report
    report_path = os.path.join(output_dir, f"{split}_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FACIAL AGE ESTIMATION — DETAILED EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Backbone: {backbone} | Head: {head_type}\n")
        f.write(f"Split: {split} | Samples: {len(target_df)}\n\n")
        f.write("OVERALL METRICS:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nPER-AGE GROUP BREAKDOWN:\n")
        f.write(bin_table.to_string(index=False))
        f.write("\n\nTOP 15 WORST PREDICTIONS:\n")
        f.write(worst_df[["filename", "age", "predicted_age", "abs_error", "error_direction"]].to_string(index=False))
    print(f"Saved evaluation report to: {report_path}")
    
    # 7. Generate Plots
    plot_predicted_vs_true(all_preds, all_targets, os.path.join(output_dir, f"{split}_pred_vs_true.png"))
    plot_error_distribution(all_preds, all_targets, os.path.join(output_dir, f"{split}_error_distribution.png"))
    plot_per_age_mae(all_preds, all_targets, os.path.join(output_dir, f"{split}_per_age_mae.png"))
    print(f"Diagnostic plots generated in: {output_dir}")
    
    return metrics, eval_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Age Estimation Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--manifest", type=str, default=None, help="Optional path to manifest.csv")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for reports/plots")
    
    args = parser.parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        split=args.split,
        output_dir=args.output_dir
    )
