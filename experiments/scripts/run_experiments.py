import os
import argparse
import pandas as pd
from datetime import datetime

from config import Config
from train import run_training

# Roadmap Experiment Configurations
EXPERIMENT_SUITES = {
    "baseline": [
        {
            "name": "exp01_baseline_effnetv2s_huber",
            "backbone": "tf_efficientnetv2_s",
            "head": "direct",
            "loss": "huber",
            "epochs": 30,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": False
        }
    ],
    "losses": [
        {
            "name": "exp02_loss_smooth_l1_huber",
            "backbone": "tf_efficientnetv2_s",
            "head": "direct",
            "loss": "huber",
            "epochs": 25,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": False
        },
        {
            "name": "exp03_loss_l1_mae",
            "backbone": "tf_efficientnetv2_s",
            "head": "direct",
            "loss": "l1",
            "epochs": 25,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": False
        },
        {
            "name": "exp04_loss_mse",
            "backbone": "tf_efficientnetv2_s",
            "head": "direct",
            "loss": "mse",
            "epochs": 25,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": False
        }
    ],
    "heads": [
        {
            "name": "exp05_head_dex_expectation",
            "backbone": "tf_efficientnetv2_s",
            "head": "dex",
            "loss": "ce",
            "epochs": 25,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": False
        },
        {
            "name": "exp06_head_ordinal_binary",
            "backbone": "tf_efficientnetv2_s",
            "head": "ordinal",
            "loss": "ordinal",
            "epochs": 25,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": False
        },
        {
            "name": "exp07_head_hybrid_fusion",
            "backbone": "tf_efficientnetv2_s",
            "head": "hybrid",
            "loss": "hybrid",
            "epochs": 25,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": False
        }
    ],
    "backbones": [
        {
            "name": "exp08_backbone_resnet50",
            "backbone": "resnet50",
            "head": "direct",
            "loss": "huber",
            "epochs": 25,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": False
        },
        {
            "name": "exp09_backbone_convnext_small",
            "backbone": "convnext_small",
            "head": "direct",
            "loss": "huber",
            "epochs": 25,
            "batch_size": 8,  # Smaller batch for ConvNeXt 4GB VRAM
            "grad_accum": 4,  # Accumulate to effective batch 32
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": False
        }
    ],
    "sampling": [
        {
            "name": "exp10_sampling_weighted_sampler",
            "backbone": "tf_efficientnetv2_s",
            "head": "direct",
            "loss": "huber",
            "epochs": 25,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": True,
            "multi_stage": False
        }
    ],
    "multistage": [
        {
            "name": "exp11_multistage_finetune",
            "backbone": "tf_efficientnetv2_s",
            "head": "direct",
            "loss": "huber",
            "epochs": 30,
            "batch_size": 16,
            "aug": "moderate",
            "weighted_sampler": False,
            "multi_stage": True
        }
    ]
}

def main():
    parser = argparse.ArgumentParser(description="Automated Age Estimation Experiment Runner")
    parser.add_argument("--suite", type=str, default="baseline", choices=list(EXPERIMENT_SUITES.keys()) + ["all"], help="Experiment suite to run")
    parser.add_argument("--quick", action="store_true", help="Quick mode (runs 2 epochs each for testing)")
    args = parser.parse_args()
    
    if args.suite == "all":
        experiments = []
        for s in EXPERIMENT_SUITES.values():
            experiments.extend(s)
    else:
        experiments = EXPERIMENT_SUITES[args.suite]
        
    leaderboard_path = os.path.join(Config().output_dir, "experiment_leaderboard.csv")
    results = []
    if os.path.exists(leaderboard_path):
        results = pd.read_csv(leaderboard_path).to_dict("records")
        
    print("=" * 70)
    print(f"FACIAL AGE ESTIMATION — EXPERIMENT RUNNER (Suite: {args.suite})")
    print(f"Total experiments to execute: {len(experiments)}")
    print("=" * 70)
    
    for idx, exp in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] Running: {exp['name']} ...")
        
        epochs = 2 if args.quick else exp["epochs"]
        grad_accum = exp.get("grad_accum", 2)
        
        cfg = Config(
            backbone=exp["backbone"],
            head_type=exp["head"],
            loss_type=exp["loss"],
            epochs=epochs,
            batch_size=exp["batch_size"],
            gradient_accumulation_steps=grad_accum,
            aug_level=exp["aug"],
            use_weighted_sampler=exp["weighted_sampler"],
            multi_stage=exp["multi_stage"]
        )
        
        start_time = datetime.now()
        try:
            metrics = run_training(cfg, exp_name=exp["name"])
            duration_mins = (datetime.now() - start_time).total_seconds() / 60.0
            
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "experiment_name": exp["name"],
                "backbone": exp["backbone"],
                "head": exp["head"],
                "loss": exp["loss"],
                "val_mae": metrics["best_val_mae"],
                "test_mae": metrics["test_mae"],
                "test_rmse": metrics["test_rmse"],
                "test_r2": metrics["test_r2"],
                "test_acc_5": metrics["test_acc_5"],
                "best_epoch": metrics["best_epoch"],
                "duration_min": round(duration_mins, 1),
                "status": "SUCCESS"
            }
        except Exception as e:
            print(f"Error in experiment {exp['name']}: {e}")
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "experiment_name": exp["name"],
                "backbone": exp["backbone"],
                "head": exp["head"],
                "loss": exp["loss"],
                "val_mae": None,
                "test_mae": None,
                "test_rmse": None,
                "test_r2": None,
                "test_acc_5": None,
                "best_epoch": None,
                "duration_min": None,
                "status": f"FAILED: {str(e)}"
            }
            
        results.append(entry)
        
        # Save Leaderboard
        lead_df = pd.DataFrame(results)
        lead_df.to_csv(leaderboard_path, index=False)
        
        print("\n--- Current Experiment Leaderboard (sorted by Test MAE) ---")
        display_df = lead_df.sort_values(by="test_mae", ascending=True)
        print(display_df[["experiment_name", "backbone", "head", "loss", "test_mae", "test_rmse", "test_acc_5", "status"]].to_string(index=False))

if __name__ == "__main__":
    main()
