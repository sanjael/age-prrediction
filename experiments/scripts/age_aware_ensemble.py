"""
age_aware_ensemble.py
End-to-End Age-Aware Hierarchical Mixture-of-Experts Ensemble
"""
import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

from age_experts import GlobalPredictionEngine, SpecialistModel, get_eval_transforms
from age_gate import AgeAwareGate

class AgeAwareEnsemble:
    """
    Hierarchical Age-Aware Mixture-of-Experts Ensemble:
    1. Global Ensemble (Model A + Model B)
    2. Age Specialists (46-60, 61-75, 76-100)
    3. Soft Age-Aware Gate
    """
    def __init__(
        self,
        model_a_path: str = "outputs/exp25_effnetv2s_dex_expected_age/best_model.pt",
        model_b_path: str = "outputs/exp23_effnetv2s_utkface_supplement/best_model.pt",
        expert_46_60_path: str = "checkpoints/expert_46_60_best.pt",
        expert_61_75_path: str = "checkpoints/expert_61_75_best.pt",
        expert_76_100_path: str = "checkpoints/expert_76_100_best.pt",
        sigma: float = 8.0,
        global_weight_min: float = 0.40,
        disagreement_scale: float = 0.0,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing AgeAwareEnsemble on {self.device}...")
        
        self.tf_orig, self.tf_flip = get_eval_transforms(320)
        
        # 1. Global Prediction Engine
        self.global_engine = GlobalPredictionEngine(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            device=device
        )
        
        # 2. Specialist Models
        self.experts = {}
        expert_configs = [
            ("46_60", (46, 60), expert_46_60_path),
            ("61_75", (61, 75), expert_61_75_path),
            ("76_100", (76, 100), expert_76_100_path),
        ]
        
        for name, rng, path in expert_configs:
            if os.path.exists(path):
                print(f" [+] Loading Expert {name} from {path}...")
                exp_model = SpecialistModel(target_range=rng, pretrained_path=path)
                exp_model.to(self.device)
                exp_model.eval()
                self.experts[name] = exp_model
            else:
                print(f" [!] Warning: Expert {name} checkpoint '{path}' not found. Fallback to base.")
                self.experts[name] = None
                
        # 3. Soft Age-Aware Gate
        self.gate = AgeAwareGate(
            sigma=sigma,
            global_weight_min=global_weight_min,
            disagreement_scale=disagreement_scale
        )
        print("[+] AgeAwareEnsemble fully initialized.")

    @torch.no_grad()
    def predict_image(self, img_pil: Image.Image) -> Dict[str, Any]:
        """
        Runs full hierarchical inference on a single PIL face image.
        Returns:
            global_age, expert_predictions, weights, final_age, inference_time_ms
        """
        start_t = time.perf_counter()
        
        t_orig = self.tf_orig(img_pil).unsqueeze(0).to(self.device)
        t_flip = self.tf_flip(img_pil).unsqueeze(0).to(self.device)
        
        # Step 1: Global Prediction & Disagreement
        pred_a, pred_b, global_pred, disagreement = self.global_engine.predict_tensor_pair(t_orig, t_flip)
        
        # Step 2: Expert Predictions (TTA)
        expert_preds = {}
        for name, expert in self.experts.items():
            if expert is not None:
                expert_preds[name] = expert.predict_tta(t_orig, t_flip)
            else:
                expert_preds[name] = global_pred  # Safe fallback to global prediction
                
        # Step 3: Dynamic Soft Gate Weights
        weights = self.gate.compute_weights(
            global_age=global_pred,
            global_disagreement=disagreement
        )
        
        # Step 4: Weighted Fusion
        final_pred = (
            weights["global"] * global_pred +
            weights["46_60"] * expert_preds["46_60"] +
            weights["61_75"] * expert_preds["61_75"] +
            weights["76_100"] * expert_preds["76_100"]
        )
        
        # Clip to valid chronological age range [1.0, 100.0]
        final_pred = float(np.clip(final_pred, 1.0, 100.0))
        
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        
        return {
            "global_age": round(global_pred, 2),
            "model_a": round(pred_a, 2),
            "model_b": round(pred_b, 2),
            "disagreement": round(disagreement, 2),
            "expert_predictions": {
                "46_60": round(expert_preds["46_60"], 2),
                "61_75": round(expert_preds["61_75"], 2),
                "76_100": round(expert_preds["76_100"], 2)
            },
            "weights": {
                "global": round(weights["global"], 4),
                "46_60": round(weights["46_60"], 4),
                "61_75": round(weights["61_75"], 4),
                "76_100": round(weights["76_100"], 4)
            },
            "final_age": round(final_pred, 2),
            "inference_time_ms": round(elapsed_ms, 2)
        }

def predict_age(image_input) -> Dict[str, Any]:
    """
    Standard unified inference API.
    Accepts image path (str) or PIL Image.
    """
    global _GLOBAL_ENSEMBLE_INSTANCE
    if "_GLOBAL_ENSEMBLE_INSTANCE" not in globals() or _GLOBAL_ENSEMBLE_INSTANCE is None:
        _GLOBAL_ENSEMBLE_INSTANCE = AgeAwareEnsemble()
        
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")
        
    return _GLOBAL_ENSEMBLE_INSTANCE.predict_image(img)
