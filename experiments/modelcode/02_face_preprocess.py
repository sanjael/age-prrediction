"""
Face Preprocessing & Alignment Engine (Phase 2 / EXP-21 to EXP-26)
-----------------------------------------------------------------
Supports 3 Preprocessing Variants:
- P0: Original Image (baseline control)
- P1: Detected Face + 10% tight margin
- P2: Detected Face + 20% contextual margin (jaw, forehead, cheeks)
- P3: 5-Point Landmark Horizontal Eye Alignment + 20% margin

Supports Custom Resolutions: 224, 256, 320, 384
"""

import os
import argparse
import math
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN


def align_and_crop_face(img_cv: np.ndarray, box: np.ndarray, landmarks: np.ndarray, 
                        variant: str = "p2", target_size: int = 320) -> np.ndarray:
    """
    Crop or align face based on variant:
    - p1: 10% bbox margin
    - p2: 20% bbox margin
    - p3: Landmark eye-horizontal alignment + 20% bbox margin
    """
    h, w = img_cv.shape[:2]
    
    if variant == "p3" and landmarks is not None and len(landmarks) >= 2:
        # Landmark eye horizontal alignment
        left_eye = landmarks[0]   # (x, y)
        right_eye = landmarks[1]  # (x, y)
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        # Re-compute bounding box with 20% margin
        margin = 0.20
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        
        crop_size = max(bw, bh) * (1.0 + 2.0 * margin)
        x1 = max(0, int(cx - crop_size / 2.0))
        y1 = max(0, int(cy - crop_size / 2.0))
        x2 = min(w, int(cx + crop_size / 2.0))
        y2 = min(h, int(cy + crop_size / 2.0))
        
        cropped = rotated[y1:y2, x1:x2]
        if cropped.size > 0:
            return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    # Standard bbox crop with margin (p1 or p2)
    margin = 0.10 if variant == "p1" else 0.20
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    
    x1 = max(0, int(box[0] - margin * bw))
    y1 = max(0, int(box[1] - margin * bh))
    x2 = min(w, int(box[2] + margin * bw))
    y2 = min(h, int(box[3] + margin * bh))
    
    cropped = img_cv[y1:y2, x1:x2]
    if cropped.size > 0:
        return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
    # Fallback to square center crop
    min_dim = min(w, h)
    x1 = (w - min_dim) // 2
    y1 = (h - min_dim) // 2
    return cv2.resize(img_cv[y1:y1+min_dim, x1:x1+min_dim], (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def process_dataset(variant: str = "p2", target_size: int = 320, max_samples: int = None, batch_size: int = 64):
    manifest_path = Path(r"e:\CTS FINAL\DATA SET\manifest_clean.csv")
    if not manifest_path.exists():
        print(f"Error: Manifest {manifest_path} not found!")
        return
        
    df = pd.read_csv(manifest_path)
    if max_samples:
        df = df.iloc[:max_samples]
        
    out_dir = Path(rf"e:\CTS FINAL\DATA SET\data_processed\faces_{variant}_{target_size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n============================================================")
    print(f"FACE PREPROCESSING: Variant={variant.upper()} | Size={target_size}x{target_size}")
    print(f"Device: {device} | Total Images to Process: {len(df):,}")
    print(f"Destination: {out_dir}")
    print(f"============================================================")
    
    detector = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)
    
    records = []
    # Process in batches for high throughput
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for b_idx in tqdm(range(num_batches), desc=f"Cropping {variant}"):
        batch_df = df.iloc[b_idx * batch_size : (b_idx + 1) * batch_size]
        pil_images = []
        cv_images = []
        dest_paths = []
        valid_indices = []
        
        for idx, row in batch_df.iterrows():
            src_p = row["filepath"]
            rel_p = Path(src_p).name
            # Preserve folder structure
            age_folder = f"{int(row['age']):03d}"
            dest_folder = out_dir / row["split"] / age_folder
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_p = dest_folder / rel_p
            
            dest_paths.append(str(dest_p))
            
            if dest_p.exists():
                # Already processed
                records.append({
                    "filepath": str(dest_p),
                    "original_filepath": src_p,
                    "filename": rel_p,
                    "age": row["age"],
                    "split": row["split"],
                    "hash": row["hash"]
                })
                continue
                
            try:
                img_cv = cv2.imread(src_p)
                if img_cv is not None:
                    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(img_rgb))
                    cv_images.append(img_cv)
                    valid_indices.append((len(dest_paths) - 1, src_p, rel_p, row["age"], row["split"], row["hash"]))
            except Exception:
                pass
                
        if len(pil_images) == 0:
            continue
            
        # Batched MTCNN detection
        try:
            boxes, _, landmarks = detector.detect(pil_images, landmarks=True)
        except Exception:
            boxes = [None] * len(pil_images)
            landmarks = [None] * len(pil_images)
            
        for i, (orig_idx, src_p, rel_p, age, split, img_hash) in enumerate(valid_indices):
            dest_p = dest_paths[orig_idx]
            img_cv = cv_images[i]
            b = boxes[i][0] if boxes is not None and boxes[i] is not None and len(boxes[i]) > 0 else None
            lm = landmarks[i][0] if landmarks is not None and landmarks[i] is not None and len(landmarks[i]) > 0 else None
            
            if b is not None:
                cropped = align_and_crop_face(img_cv, b, lm, variant=variant, target_size=target_size)
            else:
                # Center crop fallback
                h, w = img_cv.shape[:2]
                min_dim = min(h, w)
                x1 = (w - min_dim) // 2
                y1 = (h - min_dim) // 2
                cropped = cv2.resize(img_cv[y1:y1+min_dim, x1:x1+min_dim], (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                
            cv2.imwrite(dest_p, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            records.append({
                "filepath": dest_p,
                "original_filepath": src_p,
                "filename": rel_p,
                "age": age,
                "split": split,
                "hash": img_hash
            })
            
    out_manifest = Path(rf"e:\CTS FINAL\DATA SET\manifest_{variant}_{target_size}.csv")
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_manifest, index=False)
    print(f"\nCompleted {variant.upper()} preprocessing! Saved {len(out_df):,} records to {out_manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Cropping & Alignment Engine")
    parser.add_argument("--variant", type=str, default="p2", choices=["p1", "p2", "p3"], help="p1=10% margin, p2=20% margin, p3=aligned 20%")
    parser.add_argument("--size", type=int, default=320, help="Output image size (e.g. 224, 256, 320, 384)")
    parser.add_argument("--max_samples", type=int, default=None, help="Sample limit for fast validation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for MTCNN detection")
    args = parser.parse_args()
    
    process_dataset(variant=args.variant, target_size=args.size, max_samples=args.max_samples, batch_size=args.batch_size)

