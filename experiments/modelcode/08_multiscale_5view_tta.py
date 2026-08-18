import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from scipy.optimize import minimize

from models import AgeModel
from config import Config
from dataset import AgeDataset

class MultiScale5ViewTTAEvaluator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[*] Initializing 5-View Multi-Scale TTA Engine on {self.device}...")
        
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 1. View 1: Standard 320x320
        self.tf_orig = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            self.normalize
        ])
        
        # 2. View 2: Horizontal Mirror Flip
        self.tf_flip = T.Compose([
            T.Resize((320, 320)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            self.normalize
        ])
        
        # 3. View 3: 90% Center Zoom (Periorbital and fine skin wrinkles focus)
        self.tf_zoom = T.Compose([
            T.Resize((356, 356)),
            T.CenterCrop((320, 320)),
            T.ToTensor(),
            self.normalize
        ])
        
        # 4. View 4: Slight Illumination / Gamma Equalization
        self.tf_illum = T.Compose([
            T.Resize((320, 320)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            self.normalize
        ])
        
        # 5. View 5: Flipped Zoom (Cross-perspective)
        self.tf_flip_zoom = T.Compose([
            T.Resize((356, 356)),
            T.CenterCrop((320, 320)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            self.normalize
        ])
        
        self.models = {}
        self._load_models()
        
    def _load_models(self):
        # Model 1: EXP-25 DEX Champion
        p1 = "outputs/exp25_effnetv2s_dex_expected_age/best_model.pt"
        if os.path.exists(p1):
            m1 = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="dex", pretrained=False).to(self.device)
            ckpt1 = torch.load(p1, map_location=self.device)
            m1.load_state_dict(ckpt1["model_state_dict"])
            m1.eval()
            self.models["DEX_Champ"] = m1
            print(" [+] Loaded Model 1: EXP-25 (DEX 100-Way Head)")
            
        # Model 2: EXP-23 Hybrid Leader
        p2 = "outputs/exp23_effnetv2s_utkface_supplement/best_model.pt"
        if os.path.exists(p2):
            m2 = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="hybrid", pretrained=False).to(self.device)
            ckpt2 = torch.load(p2, map_location=self.device)
            m2.load_state_dict(ckpt2["model_state_dict"])
            m2.eval()
            self.models["Hybrid_Lead"] = m2
            print(" [+] Loaded Model 2: EXP-23 (Hybrid Dual Head)")
            
        # Model 3: EXP-27 ConvNeXt-Tiny (if available)
        p3 = "outputs/exp27_tri_convnext_dex_3epochs/best_model.pt"
        if os.path.exists(p3):
            m3 = AgeModel(backbone_name="convnext_tiny", head_type="dex", pretrained=False).to(self.device)
            ckpt3 = torch.load(p3, map_location=self.device)
            m3.load_state_dict(ckpt3["model_state_dict"])
            m3.eval()
            self.models["ConvNeXt_DEX"] = m3
            print(" [+] Loaded Model 3: EXP-27 (ConvNeXt-Tiny DEX)")
            
    def predict_single_image_5view(self, img_pil):
        views = [
            self.tf_orig(img_pil).unsqueeze(0).to(self.device),
            self.tf_flip(img_pil).unsqueeze(0).to(self.device),
            self.tf_zoom(img_pil).unsqueeze(0).to(self.device),
            self.tf_illum(img_pil).unsqueeze(0).to(self.device),
            self.tf_flip_zoom(img_pil).unsqueeze(0).to(self.device)
        ]
        
        preds = []
        with torch.no_grad():
            for name, model in self.models.items():
                m_view_preds = []
                for v in views:
                    if name in ["DEX_Champ", "ConvNeXt_DEX"]:
                        out = model(v)
                        probs = out["probs"][0]
                        argmax = torch.argmax(probs).item()
                        w_min = max(0, argmax - 7)
                        w_max = min(100, argmax + 8)
                        loc_p = probs[w_min:w_max] / probs[w_min:w_max].sum()
                        loc_b = torch.arange(w_min + 1, w_max + 1, device=self.device).float()
                        mode_age = torch.sum(loc_p * loc_b).item()
                        full_age = out["pred_age"].item()
                        if argmax < 7 or argmax > 80:
                            age = mode_age
                        else:
                            age = 0.6 * full_age + 0.4 * mode_age
                        m_view_preds.append(age)
                    else:
                        m_view_preds.append(model(v)["pred_age"].item())
                preds.append(np.mean(m_view_preds))
                
        # Weighted Ensemble
        if len(preds) == 2:
            p_final = 0.55 * preds[0] + 0.45 * preds[1]
        elif len(preds) >= 3:
            p_final = 0.45 * preds[0] + 0.35 * preds[1] + 0.20 * preds[2]
        else:
            p_final = preds[0]
            
        # Calibration
        p_cal = float(np.clip(1.0201 * p_final - 0.7248, 1.0, 100.0))
        return round(p_cal, 1)

if __name__ == "__main__":
    evaluator = MultiScale5ViewTTAEvaluator()
    print("[*] 5-View Multi-Scale TTA Engine initialized successfully!")
