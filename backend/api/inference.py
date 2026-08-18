"""
inference.py
Direct wrapper for GlobalDualEnsemble from predict_dual_ensemble.py.
Ensures 100% exact numerical match with local CLI script.
"""

import os
import io
import sys
import time
from PIL import Image
import torch

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from predict_dual_ensemble import GlobalDualEnsemble

class DualEnsemblePredictor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print("[*] Initializing GlobalDualEnsemble API Wrapper...")
        model_a_path = os.path.join(WORKSPACE_DIR, "outputs", "exp25_effnetv2s_dex_expected_age", "best_model.pt")
        model_b_path = os.path.join(WORKSPACE_DIR, "outputs", "exp23_effnetv2s_utkface_supplement", "best_model.pt")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ensemble = GlobalDualEnsemble(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            device=self.device
        )
        self.is_loaded = True
        print(f"[+] GlobalDualEnsemble API Wrapper active on {self.device}!\n")

    def predict_image(self, pil_image: Image.Image, use_tta: bool = True) -> dict:
        """Runs the exact GlobalDualEnsemble inference on PIL Image."""
        start_time = time.time()

        # Ensure RGB format
        img = pil_image.convert("RGB")
        
        img_o = self.ensemble.tf_orig(img).unsqueeze(0).to(self.ensemble.device)
        img_f = self.ensemble.tf_flip(img).unsqueeze(0).to(self.ensemble.device) if use_tta else img_o

        with torch.no_grad():
            # Model A (DEX Head)
            pred_a_o = self.ensemble.model_a(img_o)["pred_age"].item()
            pred_a_f = self.ensemble.model_a(img_f)["pred_age"].item() if use_tta else pred_a_o
            pred_a = 0.5 * pred_a_o + 0.5 * pred_a_f

            # Model B (Hybrid Head)
            pred_b_o = self.ensemble.model_b(img_o)["pred_age"].item()
            pred_b_f = self.ensemble.model_b(img_f)["pred_age"].item() if use_tta else pred_b_o
            pred_b = 0.5 * pred_b_o + 0.5 * pred_b_f

            # Ensemble Fusion
            ensemble_pred = 0.5 * pred_a + 0.5 * pred_b
            disagreement = abs(pred_a - pred_b)

        # Exact formula from predict_dual_ensemble.py
        ensemble_pred = ensemble_pred * 0.945
        final_age = round(float(ensemble_pred), 2)

        # Categorize
        cohort = self._get_cohort(final_age)
        latency_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "predicted_age": final_age,
            "age_group": cohort["label"],
            "cohort_code": cohort["code"],
            "model_a_dex": round(float(pred_a), 2),
            "model_b_hybrid": round(float(pred_b), 2),
            "disagreement": round(float(disagreement), 2),
            "tta_applied": use_tta,
            "bounds": {
                "pm_3": f"{max(1.0, round(final_age - 3.0, 1))} – {round(final_age + 3.0, 1)} yrs",
                "pm_5": f"{max(1.0, round(final_age - 5.0, 1))} – {round(final_age + 5.0, 1)} yrs",
                "pm_7": f"{max(1.0, round(final_age - 7.0, 1))} – {round(final_age + 7.0, 1)} yrs",
            },
            "latency_ms": latency_ms,
            "device": str(self.device),
        }

    def _get_cohort(self, age: float) -> dict:
        if age <= 12: return {"label": "👶 Child (01–12 yrs)", "code": "PED_01_12"}
        if age <= 19: return {"label": "🧑 Teen (13–19 yrs)", "code": "TEEN_13_19"}
        if age <= 35: return {"label": "👨 Young Adult (20–35 yrs)", "code": "YADULT_20_35"}
        if age <= 45: return {"label": "👩 Adult (36–45 yrs)", "code": "ADULT_36_45"}
        if age <= 60: return {"label": "👨 Middle Age (46–60 yrs)", "code": "MID_46_60"}
        if age <= 75: return {"label": "👴 Senior (61–75 yrs)", "code": "SNR_61_75"}
        return {"label": "👵 Elderly (76–100 yrs)", "code": "ELD_76_100"}
