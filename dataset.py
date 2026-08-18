import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional
from config import Config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(aug_level: str = "moderate", img_size: int = 224):
    """
    Returns train and eval torchvision transforms preserving facial structure and age cues.
    """
    eval_transform = transforms.Compose([
        transforms.Resize((int(img_size * 1.14), int(img_size * 1.14))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    if aug_level == "minimal":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    elif aug_level == "strong":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:  # "moderate" (default)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
    return train_transform, eval_transform

class AgeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, use_crop_if_avail: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_crop = use_crop_if_avail and ("filepath_cropped" in self.df.columns)
        
        # Pre-cache arrays for speed
        if self.use_crop:
            self.filepaths = self.df["filepath_cropped"].values
        else:
            self.filepaths = self.df["filepath"].values
            
        self.ages = self.df["age"].values.astype(np.float32)
        
    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fpath = self.filepaths[idx]
        age = self.ages[idx]
        
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            # Create neutral gray image if corrupt
            img = Image.new("RGB", (224, 224), (128, 128, 128))
            
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
            
        # Target representations:
        # 1. Continuous age (float32) for regression
        # 2. Discrete class index (0 to 99 for ages 1 to 100)
        age_target = torch.tensor(age, dtype=torch.float32)
        cls_target = torch.tensor(int(np.clip(age - 1, 0, 99)), dtype=torch.long)
        
        return img_tensor, age_target, cls_target

def get_weighted_sampler(df: pd.DataFrame, power: float = 0.5) -> WeightedRandomSampler:
    """
    Builds an inverse-frequency sampler to rebalance rare age groups.
    power=0.5 -> 1/sqrt(count), power=1.0 -> 1/count
    """
    age_counts = df["age"].value_counts().to_dict()
    weights_dict = {age: 1.0 / (count ** power) for age, count in age_counts.items()}
    sample_weights = df["age"].map(weights_dict).values
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def get_dataloaders(cfg: Config, manifest_df: Optional[pd.DataFrame] = None) -> dict:
    if manifest_df is None:
        if not os.path.exists(cfg.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {cfg.manifest_path}. Please run 01_data_audit.py.")
        manifest_df = pd.read_csv(cfg.manifest_path)
        
    train_df = manifest_df[manifest_df["split"] == "train"].copy()
    val_df = manifest_df[manifest_df["split"] == "val"].copy()
    test_df = manifest_df[manifest_df["split"] == "test"].copy()
    
    train_tf, eval_tf = get_transforms(aug_level=cfg.aug_level, img_size=cfg.img_size)
    
    train_dataset = AgeDataset(train_df, transform=train_tf)
    val_dataset = AgeDataset(val_df, transform=eval_tf)
    test_dataset = AgeDataset(test_df, transform=eval_tf)
    
    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        sampler = get_weighted_sampler(train_df, power=cfg.sampler_power)
        shuffle = False
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        drop_last=False
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df
    }
