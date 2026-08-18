"""
inference_age_aware.py
Single Image Inference CLI for Age-Aware Mixture-of-Experts Ensemble
"""
import os
import sys
import time
import argparse
from PIL import Image
import cv2
import numpy as np

from age_aware_ensemble import AgeAwareEnsemble

def detect_and_crop_face(img_bgr: np.ndarray) -> Image.Image:
    """
    Applies OpenCV Haar-Cascade face detection with contextual P2 margins.
    Falls back to original image if no face is detected.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad_top = int(0.25 * h)
        pad_bot = int(0.15 * h)
        pad_side = int(0.20 * w)
        
        y1 = max(0, y - pad_top)
        y2 = min(img_bgr.shape[0], y + h + pad_bot)
        x1 = max(0, x - pad_side)
        x2 = min(img_bgr.shape[1], x + w + pad_side)
        
        crop_bgr = img_bgr[y1:y2, x1:x2]
        return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    else:
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def main():
    parser = argparse.ArgumentParser(description="Age-Aware Hierarchical Mixture-of-Experts Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to input face image")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    args = parser.parse_args()
    
    img_path = args.image
    if img_path is None or not os.path.exists(img_path):
        # Pick a sample from dataset to demonstrate
        import pandas as pd
        if os.path.exists("manifest_p2_320_plus_utkface.csv"):
            df = pd.read_csv("manifest_p2_320_plus_utkface.csv")
            sample = df[df["split"] == "val"].sample(1, random_state=int(time.time()) % 1000).iloc[0]
            img_path = sample["filepath"]
            ground_truth = sample["age"]
            print(f"\n[*] No image specified. Randomly selected validation face: {img_path}")
            print(f"[*] Ground Truth Age: {ground_truth} years old\n")
        else:
            print("Error: Please provide a valid face image path using: --image <path>")
            sys.exit(1)
    else:
        ground_truth = None
        
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Could not read image from {img_path}")
        sys.exit(1)
        
    face_pil = detect_and_crop_face(img_bgr)
    
    ensemble = AgeAwareEnsemble(device=args.device)
    res = ensemble.predict_image(face_pil)
    
    print("\n" + "=" * 50)
    print("      AGE-AWARE HIERARCHICAL ENSEMBLE RESULTS")
    print("=" * 50)
    print(f"INPUT IMAGE:\n{img_path}\n")
    if ground_truth is not None:
        print(f"GROUND TRUTH AGE:\n{ground_truth} years\n")
        
    print(f"GLOBAL AGE:\n{res['global_age']:.1f}\n")
    print(f"EXPERT 46–60:\n{res['expert_predictions']['46_60']:.1f}\n")
    print(f"EXPERT 61–75:\n{res['expert_predictions']['61_75']:.1f}\n")
    print(f"EXPERT 76–100:\n{res['expert_predictions']['76_100']:.1f}\n")
    
    print("GATE WEIGHTS:\n")
    print(f"Global:\n{res['weights']['global']:.2f}\n")
    print(f"46–60:\n{res['weights']['46_60']:.2f}\n")
    print(f"61–75:\n{res['weights']['61_75']:.2f}\n")
    print(f"76–100:\n{res['weights']['76_100']:.2f}\n")
    
    print(f"FINAL AGE:\n{res['final_age']:.1f}\n")
    print(f"INFERENCE TIME:\n{res['inference_time_ms']:.1f} ms")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
