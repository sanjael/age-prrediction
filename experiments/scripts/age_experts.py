"""
age_experts.py
Specialist Age Models & Global Prediction Engine
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from models import AgeModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_eval_transforms(img_size: int = 320):
    transform_orig = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    transform_flip = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform_orig, transform_flip

class GlobalPredictionEngine:
    """
    Combines Model A (EffNetV2-S DEX) and Model B (EffNetV2-S Hybrid)
    into the baseline Global Ensemble.
    """
    def __init__(
        self,
        model_a_path: str = "outputs/exp25_effnetv2s_dex_expected_age/best_model.pt",
        model_b_path: str = "outputs/exp23_effnetv2s_utkface_supplement/best_model.pt",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tf_orig, self.tf_flip = get_eval_transforms(320)
        
        # Load Model A (DEX Head)
        print(f"[*] Loading Model A from {model_a_path}...")
        self.model_a = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="dex", pretrained=False).to(self.device)
        ckpt_a = torch.load(model_a_path, map_location=self.device)
        self.model_a.load_state_dict(ckpt_a["model_state_dict"])
        self.model_a.eval()
        
        # Load Model B (Hybrid Head)
        print(f"[*] Loading Model B from {model_b_path}...")
        self.model_b = AgeModel(backbone_name="tf_efficientnetv2_s", head_type="hybrid", pretrained=False).to(self.device)
        ckpt_b = torch.load(model_b_path, map_location=self.device)
        self.model_b.load_state_dict(ckpt_b["model_state_dict"])
        self.model_b.eval()
        print("[+] Global Models A and B loaded successfully.")

    @torch.no_grad()
    def predict_tensor_pair(self, t_orig: torch.Tensor, t_flip: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        Runs 2-view TTA on Model A and Model B.
        Returns: (pred_a, pred_b, global_pred, disagreement)
        """
        # Model A (DEX)
        out_a_orig = self.model_a(t_orig)["pred_age"].item()
        out_a_flip = self.model_a(t_flip)["pred_age"].item()
        pred_a = 0.5 * out_a_orig + 0.5 * out_a_flip
        
        # Model B (Hybrid)
        out_b_orig = self.model_b(t_orig)["pred_age"].item()
        out_b_flip = self.model_b(t_flip)["pred_age"].item()
        pred_b = 0.5 * out_b_orig + 0.5 * out_b_flip
        
        # Global Ensemble: 50/50 mean
        global_pred = 0.5 * pred_a + 0.5 * pred_b
        disagreement = abs(pred_a - pred_b)
        
        return pred_a, pred_b, global_pred, disagreement

    def predict_image(self, img_pil: Image.Image) -> Dict[str, float]:
        t_orig = self.tf_orig(img_pil).unsqueeze(0).to(self.device)
        t_flip = self.tf_flip(img_pil).unsqueeze(0).to(self.device)
        pred_a, pred_b, global_pred, disagreement = self.predict_tensor_pair(t_orig, t_flip)
        return {
            "pred_a": pred_a,
            "pred_b": pred_b,
            "global_prediction": global_pred,
            "global_disagreement": disagreement
        }

class SpecialistModel(nn.Module):
    """
    Age-Specialist Model built on pretrained EfficientNetV2-S with DEX head.
    Specialized for a target age sub-domain (e.g. 46-60, 61-75, 76-100).
    """
    def __init__(
        self,
        target_range: Tuple[int, int],
        backbone_name: str = "tf_efficientnetv2_s",
        pretrained_path: Optional[str] = "outputs/exp25_effnetv2s_dex_expected_age/best_model.pt",
        drop_rate: float = 0.2
    ):
        super().__init__()
        self.target_range = target_range
        self.model = AgeModel(backbone_name=backbone_name, head_type="dex", pretrained=False, drop_rate=drop_rate)
        
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[*] Initializing Specialist {target_range[0]}-{target_range[1]} from {pretrained_path}...")
            ckpt = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[+] Transfer weights loaded for Specialist {target_range[0]}-{target_range[1]}.")
        else:
            print(f"[!] Warning: Pretrained path not found, initialized from scratch.")
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    @torch.no_grad()
    def predict_tta(self, t_orig: torch.Tensor, t_flip: torch.Tensor) -> float:
        out_orig = self.model(t_orig)["pred_age"].item()
        out_flip = self.model(t_flip)["pred_age"].item()
        return 0.5 * out_orig + 0.5 * out_flip
