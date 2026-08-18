"""
age_gate.py
Soft Age-Aware Gating Mechanism for Hierarchical Mixture-of-Experts
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

class AgeAwareGate:
    """
    Soft Age-Aware Gate:
    Calculates dynamic gating weights for the Global Model and the 3 Age Specialists
    (46-60, 61-75, 76-100) based on the global age prediction and optional disagreement.
    """
    def __init__(
        self,
        center_46_60: float = 53.0,
        center_61_75: float = 68.0,
        center_76_100: float = 88.0,
        sigma: float = 8.0,
        global_weight_min: float = 0.40,
        disagreement_scale: float = 0.0
    ):
        self.centers = {
            "46_60": center_46_60,
            "61_75": center_61_75,
            "76_100": center_76_100
        }
        self.sigma = sigma
        self.global_weight_min = global_weight_min
        self.disagreement_scale = disagreement_scale

    def compute_weights(
        self,
        global_age: float,
        global_disagreement: float = 0.0
    ) -> Dict[str, float]:
        """
        Computes normalized non-negative weights summing to 1.0:
        w_global, w_46_60, w_61_75, w_76_100
        """
        # 1. RBF proximity to each expert center
        raw_rbf = {}
        for name, center in self.centers.items():
            dist_sq = (global_age - center) ** 2
            raw_rbf[name] = np.exp(-dist_sq / (2.0 * (self.sigma ** 2)))
            
        total_rbf = sum(raw_rbf.values())
        
        # If global age is far from all specialists (e.g. young children/adults < 40),
        # specialists should have near-zero weight and global model has ~100% weight.
        # Max RBF peak is 1.0.
        max_rbf = max(raw_rbf.values())
        
        # Effective specialist capacity budget increases when near older regions
        # If global_age < 40, max_rbf < 0.05 -> specialists get almost 0.
        specialist_budget = (1.0 - self.global_weight_min) * min(1.0, max_rbf * 1.2)
        
        # Optional adjustment based on model disagreement
        if self.disagreement_scale > 0.0 and global_disagreement > 0.0:
            # High disagreement -> shift slightly more budget to specialist
            boost = min(0.20, (global_disagreement / 20.0) * self.disagreement_scale)
            specialist_budget = min(1.0 - self.global_weight_min * 0.5, specialist_budget + boost)
            
        w_global = 1.0 - specialist_budget
        
        if total_rbf > 1e-6:
            w_46_60 = specialist_budget * (raw_rbf["46_60"] / total_rbf)
            w_61_75 = specialist_budget * (raw_rbf["61_75"] / total_rbf)
            w_76_100 = specialist_budget * (raw_rbf["76_100"] / total_rbf)
        else:
            w_46_60 = 0.0
            w_61_75 = 0.0
            w_76_100 = 0.0
            w_global = 1.0
            
        # Ensure non-negativity and exact sum to 1.0
        weights = {
            "global": float(max(0.0, w_global)),
            "46_60": float(max(0.0, w_46_60)),
            "61_75": float(max(0.0, w_61_75)),
            "76_100": float(max(0.0, w_76_100))
        }
        total_w = sum(weights.values())
        for k in weights:
            weights[k] /= total_w
            
        return weights
