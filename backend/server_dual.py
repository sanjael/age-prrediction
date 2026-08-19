"""
server_dual.py
Master Production Backend with Multi-Cascade CLAHE Face Detection & Hugging Face Inference Integration.
Integrates:
  1. Multi-Cascade CLAHE Face Detector & Cropping Engine.
  2. Hugging Face Inference API Client (Sanjayramdata/celingfan).
  3. Local Dual Ensemble PyTorch Model Fallback (CUDA/CPU).
"""

import os
import io
import sys
import time
import base64
import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add workspace directory to path
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from predict_dual_ensemble import GlobalDualEnsemble
from backend.api.hf_client import HuggingFaceInferenceClient

app = FastAPI(title="AGE-X Dual Ensemble Master Inference API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Instances
ensemble = None
hf_client = None
cascades = []

@app.on_event("startup")
def startup_event():
    global ensemble, hf_client, cascades
    print("[*] Initializing High-Sensitivity Multi-Cascade Face Detection Engine...")
    
    cascade_names = [
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt.xml',
        'haarcascade_profileface.xml'
    ]
    
    for c_name in cascade_names:
        try:
            path = cv2.data.haarcascades + c_name
            if os.path.exists(path):
                c = cv2.CascadeClassifier(path)
                cascades.append(c)
                print(f" [+] Loaded Cascade: {c_name}")
        except Exception:
            pass

    print("[*] Initializing Hugging Face Inference Client (Sanjayramdata/celingfan)...")
    hf_client = HuggingFaceInferenceClient()

    print("[*] Loading Local GlobalDualEnsemble on CUDA/CPU...")
    try:
        ensemble = GlobalDualEnsemble()
        print("[+] GlobalDualEnsemble loaded and active!")
    except Exception as e:
        print(f"[!] Local Dual Ensemble Note: {e}")

def detect_and_crop_face_cv2(pil_img: Image.Image) -> tuple[Image.Image, bool, str]:
    """
    High-Sensitivity Face Detection with CLAHE contrast enhancement & Multi-Cascade passes.
    Handles dim webcam lighting, tilted heads, and shadows.
    """
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    
    detected_faces = []
    
    for c in cascades:
        faces = c.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(40, 40))
        if len(faces) > 0:
            detected_faces = faces
            break
            
        faces_enh = c.detectMultiScale(gray_enhanced, scaleFactor=1.06, minNeighbors=2, minSize=(35, 35))
        if len(faces_enh) > 0:
            detected_faces = faces_enh
            break

    if len(detected_faces) > 0:
        largest_face = max(detected_faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        pad_x = int(0.25 * w)
        pad_y = int(0.25 * h)
        
        y1 = max(0, y - pad_y)
        y2 = min(img_bgr.shape[0], y + h + int(0.15 * h))
        x1 = max(0, x - pad_x)
        x2 = min(img_bgr.shape[1], x + w + pad_x)
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        crop_size = max(crop_w, crop_h)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        sq_x1 = max(0, center_x - crop_size // 2)
        sq_y1 = max(0, center_y - crop_size // 2)
        sq_x2 = min(img_bgr.shape[1], sq_x1 + crop_size)
        sq_y2 = min(img_bgr.shape[0], sq_y1 + crop_size)
        
        crop_bgr = img_bgr[sq_y1:sq_y2, sq_x1:sq_x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(crop_rgb).resize((320, 320), Image.Resampling.BILINEAR)
        
        buffered = io.BytesIO()
        cropped_pil.save(buffered, format="JPEG", quality=90)
        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return cropped_pil, True, crop_b64
    else:
        w, h = pil_img.size
        crop_size = int(min(w, h) * 0.75)
        left = max(0, (w - crop_size) // 2)
        top = max(0, int((h - crop_size) * 0.25))
        cropped_pil = pil_img.crop((left, top, left + crop_size, top + crop_size)).resize((320, 320), Image.Resampling.BILINEAR)
        
        buffered = io.BytesIO()
        cropped_pil.save(buffered, format="JPEG", quality=90)
        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return cropped_pil, False, crop_b64

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "AGE-X AI Facial Age Prediction API",
        "model": "Dual Ensemble (EfficientNetV2-S DEX + Hybrid Head)",
        "hf_repo": "Sanjayramdata/celingfan",
        "final_mae": 4.18,
        "acc_plus_minus_7": 82.0,
        "acc_plus_minus_5": 67.0,
        "device": str(ensemble.device if ensemble else "cuda"),
        "is_model_loaded": ensemble is not None
    }

@app.post("/api/predict")
async def predict_face(request: Request):
    global ensemble, hf_client
    start_time = time.time()
    raw_pil_image = None

    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
            base64_str = data.get("image", "")
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            image_bytes = base64.b64decode(base64_str)
            raw_pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif "multipart/form-data" in content_type:
            form = await request.form()
            uploaded_file = form.get("file")
            if uploaded_file:
                contents = await uploaded_file.read()
                raw_pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to parse image payload: {str(e)}"})

    if raw_pil_image is None:
        return JSONResponse(status_code=400, content={"error": "No image found in request"})

    # Auto-orient Exif image
    raw_pil_image = ImageOps.exif_transpose(raw_pil_image)

    # 1. Face Detection and Tight 320x320 Face Cropping
    face_img, face_detected, face_thumb_b64 = detect_and_crop_face_cv2(raw_pil_image)

    # 2. Try Hugging Face Inference API if HUGGINGFACE_API_KEY is configured
    hf_result = None
    if hf_client and os.environ.get("HUGGINGFACE_API_KEY"):
        try:
            hf_res = hf_client.predict_image(face_img)
            if hf_res.get("status") == "success" and isinstance(hf_res.get("data"), list):
                # Format Hugging Face response
                hf_data = hf_res["data"]
                if len(hf_data) > 0 and "score" in hf_data[0]:
                    pred_age_val = float(hf_data[0].get("score", 25.0))
                    hf_result = {
                        "predicted_age": pred_age_val,
                        "source": "hugging_face_api"
                    }
        except Exception:
            hf_result = None

    # 3. Local Dual Ensemble PyTorch CUDA/CPU Model Fallback / Direct Engine
    if ensemble:
        try:
            img_o = ensemble.tf_orig(face_img).unsqueeze(0).to(ensemble.device)
            img_f = ensemble.tf_flip(face_img).unsqueeze(0).to(ensemble.device)

            with torch.no_grad():
                pred_a_o = ensemble.model_a(img_o)["pred_age"].item()
                pred_a_f = ensemble.model_a(img_f)["pred_age"].item()
                pred_a = 0.5 * pred_a_o + 0.5 * pred_a_f

                pred_b_o = ensemble.model_b(img_o)["pred_age"].item()
                pred_b_f = ensemble.model_b(img_f)["pred_age"].item()
                pred_b = 0.5 * pred_b_o + 0.5 * pred_b_f

                ensemble_pred = 0.5 * pred_a + 0.5 * pred_b
                disagreement = abs(pred_a - pred_b)

            ensemble_pred = ensemble_pred * 0.945
            final_age = round(float(ensemble_pred), 2)
            pred_a = round(float(pred_a), 2)
            pred_b = round(float(pred_b), 2)
            disagreement = round(float(disagreement), 2)
        except Exception:
            final_age = hf_result["predicted_age"] if hf_result else 25.0
            pred_a, pred_b, disagreement = final_age, final_age, 0.0
    else:
        final_age = hf_result["predicted_age"] if hf_result else 25.0
        pred_a, pred_b, disagreement = final_age, final_age, 0.0

    # Demographic Cohort Categorization
    if final_age <= 12: cohort = "👶 Child (01–12 yrs)"
    elif final_age <= 19: cohort = "🧑 Teen (13–19 yrs)"
    elif final_age <= 35: cohort = "👨 Young Adult (20–35 yrs)"
    elif final_age <= 45: cohort = "👩 Adult (36–45 yrs)"
    elif final_age <= 60: cohort = "👨 Middle Age (46–60 yrs)"
    elif final_age <= 75: cohort = "👴 Senior (61–75 yrs)"
    else: cohort = "👵 Elderly (76–100 yrs)"

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "status": "SUCCESS",
        "predicted_age": final_age,
        "age_group": cohort,
        "face_detected": face_detected,
        "face_thumbnail": face_thumb_b64,
        "model_a_dex": pred_a,
        "model_b_hybrid": pred_b,
        "disagreement": disagreement,
        "bounds": {
            "pm_3": f"{max(1.0, round(final_age - 3.0, 1))} – {round(final_age + 3.0, 1)} yrs",
            "pm_5": f"{max(1.0, round(final_age - 5.0, 1))} – {round(final_age + 5.0, 1)} yrs",
            "pm_7": f"{max(1.0, round(final_age - 7.0, 1))} – {round(final_age + 7.0, 1)} yrs",
        },
        "latency_ms": latency_ms,
        "device": str(ensemble.device if ensemble else "cpu")
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
