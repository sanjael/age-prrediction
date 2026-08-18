import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DirectHead(nn.Module):
    """
    Continuous regression head predicting scalar age directly.
    """
    def __init__(self, in_features: int, drop_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x: torch.Tensor) -> dict:
        pred_age = self.net(x).squeeze(-1)
        return {
            "pred_age": pred_age,
            "raw_output": pred_age
        }

class DEXHead(nn.Module):
    """
    Deep EXpectation (DEX) Head: 100-way classification over discrete ages 1..100.
    Expected age = sum_{i=1}^100 (p_i * i)
    """
    def __init__(self, in_features: int, num_classes: int = 100, drop_rate: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        self.register_buffer("age_bins", torch.arange(1, num_classes + 1, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> dict:
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        expected_age = torch.sum(probs * self.age_bins, dim=-1)
        return {
            "pred_age": expected_age,
            "logits": logits,
            "probs": probs
        }

class OrdinalHead(nn.Module):
    """
    Ordinal Regression Head with 99 binary classification thresholds (P(Age > k)).
    Age = 1 + sum_{k=1}^99 sigmoid(z_k)
    """
    def __init__(self, in_features: int, num_classes: int = 100, drop_rate: float = 0.2):
        super().__init__()
        self.num_thresholds = num_classes - 1  # 99 thresholds for ages 1..100
        self.net = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, self.num_thresholds)
        )
        
    def forward(self, x: torch.Tensor) -> dict:
        logits = self.net(x)  # [B, 99]
        probs = torch.sigmoid(logits)
        pred_age = 1.0 + torch.sum(probs, dim=-1)
        return {
            "pred_age": pred_age,
            "logits": logits,
            "probs": probs
        }

class HybridHead(nn.Module):
    """
    Multi-task hybrid head combining Direct Regression + DEX Expectation with learnable fusion.
    """
    def __init__(self, in_features: int, num_classes: int = 100, drop_rate: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.direct_branch = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        self.dex_branch = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, num_classes)
        )
        # Learnable fusion parameter
        self.fusion_logit = nn.Parameter(torch.tensor([0.0]))  # sigmoid(0.0) = 0.5
        self.register_buffer("age_bins", torch.arange(1, num_classes + 1, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> dict:
        reg_age = self.direct_branch(x).squeeze(-1)
        
        logits = self.dex_branch(x)
        probs = F.softmax(logits, dim=-1)
        dex_age = torch.sum(probs * self.age_bins, dim=-1)
        
        weight = torch.sigmoid(self.fusion_logit)
        pred_age = weight * reg_age + (1.0 - weight) * dex_age
        
        return {
            "pred_age": pred_age,
            "reg_age": reg_age,
            "dex_age": dex_age,
            "logits": logits,
            "probs": probs,
            "fusion_weight": weight
        }

class AgeModel(nn.Module):
    """
    Complete Facial Age Estimation Model with timm backbone and selectable head architecture.
    """
    def __init__(
        self,
        backbone_name: str = "tf_efficientnetv2_s",
        head_type: str = "direct",
        pretrained: bool = True,
        num_classes: int = 100,
        drop_rate: float = 0.2
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.head_type = head_type.lower()
        self.num_classes = num_classes
        
        # Load backbone without classification head (features only)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0  # Returns pooled feature vector [B, in_features]
        )
        
        in_features = self.backbone.num_features
        
        # Instantiate requested head
        if self.head_type == "direct":
            self.head = DirectHead(in_features, drop_rate=drop_rate)
        elif self.head_type == "dex":
            self.head = DEXHead(in_features, num_classes=num_classes, drop_rate=drop_rate)
        elif self.head_type == "ordinal":
            self.head = OrdinalHead(in_features, num_classes=num_classes, drop_rate=drop_rate)
        elif self.head_type == "hybrid":
            self.head = HybridHead(in_features, num_classes=num_classes, drop_rate=drop_rate)
        else:
            raise ValueError(f"Unknown head type: '{head_type}'. Choose from direct, dex, ordinal, hybrid.")
            
    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        out = self.head(features)
        return out

    def freeze_backbone(self, freeze: bool = True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
            
    def unfreeze_last_n_blocks(self, n: int = 2):
        # Freeze all first
        self.freeze_backbone(True)
        
        # Unfreeze last parameters
        params = list(self.backbone.parameters())
        num_to_unfreeze = int(len(params) * (n / 10.0))
        for p in params[-max(num_to_unfreeze, 10):]:
            p.requires_grad = True

    def get_parameter_groups(self, head_lr: float, backbone_lr: float, weight_decay: float = 1e-2):
        decay_backbone = []
        no_decay_backbone = []
        decay_head = []
        no_decay_head = []
        
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name:
                no_decay_backbone.append(param)
            else:
                decay_backbone.append(param)
                
        for name, param in self.head.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name:
                no_decay_head.append(param)
            else:
                decay_head.append(param)
                
        param_groups = [
            {"params": decay_backbone, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": no_decay_backbone, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": decay_head, "lr": head_lr, "weight_decay": weight_decay},
            {"params": no_decay_head, "lr": head_lr, "weight_decay": 0.0},
        ]
        return [g for g in param_groups if len(g["params"]) > 0]
