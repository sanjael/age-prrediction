import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import Config
from models import AgeModel
from dataset import IMAGENET_MEAN, IMAGENET_STD

def get_tta_transforms(img_size: int = 224):
    """
    Returns list of test-time augmentation transforms:
    1. Standard CenterCrop
    2. Horizontal Flip
    3. Slight Scale Up + CenterCrop
    4. Slight Scale Down + CenterCrop
    """
    base_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    t1 = transforms.Compose([
        transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        base_norm
    ])
    
    t2 = transforms.Compose([
        transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        base_norm
    ])
    
    t3 = transforms.Compose([
        transforms.Resize((int(img_size * 1.20), int(img_size * 1.20))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        base_norm
    ])
    
    return [t1, t2, t3]

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> AgeModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg_dict = checkpoint.get("config", {})
    backbone = cfg_dict.get("backbone", "tf_efficientnetv2_s")
    head_type = cfg_dict.get("head_type", "direct")
    num_classes = cfg_dict.get("num_classes", 100)
    
    model = AgeModel(
        backbone_name=backbone,
        head_type=head_type,
        pretrained=False,
        num_classes=num_classes
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict_single_image(img_path: str, models: list, use_tta: bool = True, device: torch.device = None) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return 30.0  # Default median fallback
        
    tta_transforms = get_tta_transforms() if use_tta else [get_tta_transforms()[0]]
    
    all_preds = []
    with torch.no_grad():
        for tf in tta_transforms:
            tensor = tf(img).unsqueeze(0).to(device)
            for model in models:
                out = model(tensor)
                pred = out["pred_age"].item()
                all_preds.append(pred)
                
    return float(np.mean(all_preds))

def run_submission_inference(
    checkpoints: list,
    manifest_path: str,
    output_csv: str = "submission.csv",
    use_tta: bool = True,
    split: str = "test"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {len(checkpoints)} model(s) for ensemble inference on device {device}...")
    
    models = [load_model_from_checkpoint(cp, device) for cp in checkpoints]
    
    df = pd.read_csv(manifest_path)
    if split:
        df = df[df["split"] == split].reset_index(drop=True)
        
    print(f"Generating predictions for {len(df):,} images (TTA={use_tta})...")
    
    tta_transforms = get_tta_transforms() if use_tta else [get_tta_transforms()[0]]
    preds = []
    
    # Process images
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        fpath = row["filepath"]
        pred_age = predict_single_image(fpath, models, use_tta=use_tta, device=device)
        preds.append(round(pred_age, 2))
        
    df["predicted_age"] = preds
    if "age" in df.columns:
        df["abs_error"] = np.abs(df["predicted_age"] - df["age"])
        mae = df["abs_error"].mean()
        print(f"\nFinal Test MAE with Ensemble/TTA: {mae:.2f} years")
        
    submission_df = df[["filename", "predicted_age"]].rename(columns={"filename": "Id", "predicted_age": "PredictedAge"})
    submission_df.to_csv(output_csv, index=False)
    print(f"Submission saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test-Time Augmentation & Ensemble Inference")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint paths for ensembling")
    parser.add_argument("--manifest", type=str, default=r"e:\CTS FINAL\DATA SET\manifest_clean.csv", help="Path to manifest")
    parser.add_argument("--image", type=str, default=None, help="Predict on a single image file")
    parser.add_argument("--output", type=str, default=r"e:\CTS FINAL\DATA SET\submission.csv", help="Output CSV path")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--split", type=str, default="test", help="Manifest split")
    
    args = parser.parse_args()
    
    if args.image:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = [load_model_from_checkpoint(cp, device) for cp in args.checkpoints]
        pred = predict_single_image(args.image, models, use_tta=not args.no_tta, device=device)
        print(f"\nPredicted Age for {args.image}: {pred:.2f} years")
    else:
        run_submission_inference(
            checkpoints=args.checkpoints,
            manifest_path=args.manifest,
            output_csv=args.output,
            use_tta=not args.no_tta,
            split=args.split
        )
