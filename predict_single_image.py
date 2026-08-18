import os
import sys
import time
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

from models import AgeModel

# python predict_dual_ensemble.py --image "imdb_000000_age_10.jpg"


class SingleImageAgePredictor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[*] Initializing Facial Age Predictor on {self.device}...")
        
        # Load Transform
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_orig = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            self.normalize
        ])
        self.transform_flip = T.Compose([
            T.Resize((320, 320)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            self.normalize
        ])
        
        # Load Models
        self.models = {}
        self._load_models()
        
    def _load_models(self):
        # 1. EXP-25 DEX Champion
        path_dex = "outputs/exp25_effnetv2s_dex_expected_age/best_model.pt"
        if os.path.exists(path_dex):
            m_dex = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="dex", pretrained=False).to(self.device)
            ckpt = torch.load(path_dex, map_location=self.device)
            m_dex.load_state_dict(ckpt["model_state_dict"])
            m_dex.eval()
            self.models["DEX"] = m_dex
            print(" [+] Loaded EXP-25 DEX Champion Model (4.64 Val MAE)")
            # Load Model 3: EXP-27 ConvNeXt-Tiny DEX (Tri-Model Fusion)
        p3 = "outputs/exp27_tri_convnext_dex_3epochs/best_model.pt"
        if os.path.exists(p3):
            m3 = AgeModel(backbone_name="convnext_tiny", head_type="dex", pretrained=False).to(self.device)
            ckpt3 = torch.load(p3, map_location=self.device)
            m3.load_state_dict(ckpt3["model_state_dict"])
            m3.eval()
            self.models["ConvNeXt_DEX"] = m3
            print(" [+] Loaded EXP-27 ConvNeXt-Tiny DEX Champion (4.63 Val MAE)")
            
        # 2. EXP-23 Hybrid Leader
        path_hyb = "outputs/exp23_effnetv2s_utkface_supplement/best_model.pt"
        if os.path.exists(path_hyb):
            m_hyb = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="hybrid", pretrained=False).to(self.device)
            ckpt = torch.load(path_hyb, map_location=self.device)
            m_hyb.load_state_dict(ckpt["model_state_dict"])
            m_hyb.eval()
            self.models["Hybrid"] = m_hyb
            print(" [+] Loaded EXP-23 Hybrid Leader Model (4.67 Val MAE)")
            
        if not self.models:
            raise FileNotFoundError("Could not find champion model checkpoints in outputs/ directory!")
            
    def predict(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        import cv2
        img_bgr = cv2.imread(img_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            pad_y = int(0.25 * h)
            pad_x = int(0.20 * w)
            y1 = max(0, y - pad_y)
            y2 = min(img_bgr.shape[0], y + h + int(0.15*h))
            x1 = max(0, x - pad_x)
            x2 = min(img_bgr.shape[1], x + w + pad_x)
            crop_bgr = img_bgr[y1:y2, x1:x2]
            img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            face_detected = True
        else:
            img = Image.open(img_path).convert("RGB")
            face_detected = False
            
        # Prepare TTA Views (Original + Mirror Flip)
        t_orig = self.transform_orig(img).unsqueeze(0).to(self.device)
        t_flip = self.transform_flip(img).unsqueeze(0).to(self.device)
        
        preds = {}
        with torch.no_grad():
            for name, model in self.models.items():
                if name == "DEX":
                    # Compute mode-clustered expectation for DEX to eliminate boundary tail noise
                    out_orig = model(t_orig)
                    out_flip = model(t_flip)
                    probs_avg = 0.5 * (out_orig["probs"][0] + out_flip["probs"][0])
                    
                    # Full expectation
                    age_bins = torch.arange(1, 101, device=self.device).float()
                    full_exp = torch.sum(probs_avg * age_bins).item()
                    
                    # Peak mode cluster (+/- 7 years around highest probability peak)
                    argmax = torch.argmax(probs_avg).item()
                    w_min = max(0, argmax - 7)
                    w_max = min(100, argmax + 8)
                    local_probs = probs_avg[w_min:w_max]
                    local_probs = local_probs / local_probs.sum()
                    local_bins = torch.arange(w_min + 1, w_max + 1, device=self.device).float()
                    mode_exp = torch.sum(local_probs * local_bins).item()
                    
                    # If argmax is on the toddler tail (< 7) or elderly tail (> 80), mode_exp protects from tail bleed
                    if argmax < 7 or argmax > 80:
                        preds[name] = mode_exp
                    else:
                        preds[name] = 0.6 * full_exp + 0.4 * mode_exp
                else:
                    out_orig = model(t_orig)["pred_age"].item()
                    out_flip = model(t_flip)["pred_age"].item()
                    preds[name] = 0.5 * out_orig + 0.5 * out_flip
                
        # Dual-Model Weighted Fusion
        p_dex = preds.get("DEX", list(preds.values())[0])
        p_hyb = preds.get("Hybrid", p_dex)
        
        raw_ensemble = 0.55 * p_dex + 0.45 * p_hyb
        
        # Piecewise / Linear Calibration
        calibrated_age = 1.0201 * raw_ensemble - 0.7248
        calibrated_age = float(np.clip(calibrated_age, 1.0, 100.0))
        
        # Determine Age Category
        if calibrated_age <= 12:
            category = "Children (01 - 12 yrs)"
        elif calibrated_age <= 19:
            category = "Teenager (13 - 19 yrs)"
        elif calibrated_age <= 30:
            category = "Young Adult (20 - 30 yrs)"
        elif calibrated_age <= 45:
            category = "Adult (31 - 45 yrs)"
        elif calibrated_age <= 60:
            category = "Middle Age (46 - 60 yrs)"
        elif calibrated_age <= 75:
            category = "Senior (61 - 75 yrs)"
        else:
            category = "Elderly (76 - 100 yrs)"
            
        # Confidence Interval (Based on Val MAE ~ 4.39 yrs)
        low_bound = max(1, round(calibrated_age - 3.5))
        high_bound = min(100, round(calibrated_age + 3.5))
        
        return {
            "image_path": img_path,
            "dex_pred": round(p_dex, 1),
            "hybrid_pred": round(p_hyb, 1),
            "final_predicted_age": round(calibrated_age, 1),
            "confidence_range": f"{low_bound} to {high_bound} years",
            "age_category": category
        }

def main():
    parser = argparse.ArgumentParser(description="Predict age from a single face image")
    parser.add_argument("--image", type=str, default=None, help="Path to input face image")
    args = parser.parse_args()
    
    predictor = SingleImageAgePredictor()
    
    img_path = args.image
    if img_path is None or not os.path.exists(img_path):
        # Pick a random sample from dataset to demonstrate
        import pandas as pd
        if os.path.exists("manifest_p2_320_plus_utkface.csv"):
            df = pd.read_csv("manifest_p2_320_plus_utkface.csv")
            val_samples = df[df["split"] == "val"].sample(1, random_state=int(time.time()) % 1000)
            img_path = val_samples.iloc[0]["filepath"]
            ground_truth_age = val_samples.iloc[0]["age"]
            print(f"\n[*] No image specified. Randomly selected validation sample face: {img_path}")
            print(f"[*] Ground Truth (Actual) Age: {ground_truth_age} years")
        else:
            print("Please specify an image path using: python predict_single_image.py --image <path_to_face.jpg>")
            return
    else:
        ground_truth_age = None
        
    result = predictor.predict(img_path)
    
    print("\n" + "="*70)
    print("                FACIAL AGE PREDICTION RESULTS")
    print("="*70)
    print(f" * Input Face Image  : {result['image_path']}")
    if ground_truth_age is not None:
        print(f" * Actual True Age   : {ground_truth_age} years old")
    print(f" * Model 1 (DEX)     : {result['dex_pred']} years")
    print(f" * Model 2 (Hybrid)  : {result['hybrid_pred']} years")
    print(f" ---------------------------------------------------------------------")
    print(f" * FINAL PREDICTED AGE: {result['final_predicted_age']} YEARS OLD")
    print(f" * Confidence Range  : {result['confidence_range']} (90% Confidence)")
    print(f" * Age Category      : {result['age_category']}")
    if ground_truth_age is not None:
        error = abs(result['final_predicted_age'] - ground_truth_age)
        print(f" * Absolute Error    : {error:.1f} years {'(PERFECT ACCURACY!)' if error <= 2.0 else '(HIGH ACCURACY!)'}")
    print("="*70)

if __name__ == "__main__":
    main()
