"""
server_dual.py
Master Production Backend with OpenCV/YOLO-Style Face Detection & Cropping Pipeline.
Directly uses GlobalDualEnsemble from predict_dual_ensemble.py on CUDA GPU.
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

# Add workspace directory to path
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from predict_dual_ensemble import GlobalDualEnsemble

app = FastAPI(title="AGE-X Dual Ensemble Master Inference API", version="3.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance & Face Detector
ensemble = None
face_cascade = None
profile_cascade = None

@app.on_event("startup")
def startup_event():
    global ensemble, face_cascade, profile_cascade
    print("[*] Initializing Face Detection Neural Cascade...")
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        print("[+] Face Detection Cascade loaded successfully!")
    except Exception as e:
        print(f"[!] Warning loading face cascade: {e}")

    print("[*] Loading GlobalDualEnsemble on CUDA...")
    ensemble = GlobalDualEnsemble()
    print("[+] GlobalDualEnsemble loaded and active on port 8000!\n")

def detect_and_crop_face_cv2(pil_img: Image.Image) -> tuple[Image.Image, bool, str]:
    """
    Detects human face in image, adds 20% margin, and extracts square 320x320 crop.
    Returns: (cropped_pil_image, face_detected_bool, base64_cropped_thumbnail)
    """
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Run multi-scale face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        # Try profile cascade for angled faces
        faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(60, 60)
        )

    if len(faces) > 0:
        # Pick largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Add 25% padding around face to capture complete chin & cranial boundary
        pad_x = int(0.25 * w)
        pad_y = int(0.25 * h)
        
        y1 = max(0, y - pad_y)
        y2 = min(img_bgr.shape[0], y + h + int(0.15 * h))
        x1 = max(0, x - pad_x)
        x2 = min(img_bgr.shape[1], x + w + pad_x)
        
        # Make crop square
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
        
        # Generate base64 thumbnail of the detected face
        buffered = io.BytesIO()
        cropped_pil.save(buffered, format="JPEG", quality=90)
        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return cropped_pil, True, crop_b64
    else:
        # Fallback: if already a tight 1:1 face crop (e.g. UTKFace/IMDB image), use center square
        w, h = pil_img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        cropped_pil = pil_img.crop((left, top, left + min_dim, top + min_dim)).resize((320, 320), Image.Resampling.BILINEAR)
        
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
        "final_mae": 4.18,
        "acc_plus_minus_7": 82.0,
        "acc_plus_minus_5": 67.0,
        "device": str(ensemble.device if ensemble else "cuda"),
        "is_model_loaded": ensemble is not None
    }

@app.post("/api/predict")
async def predict_face(request: Request):
    global ensemble
    if ensemble is None:
        ensemble = GlobalDualEnsemble()

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

    # 2. Run GlobalDualEnsemble on the cropped face
    try:
        img_o = ensemble.tf_orig(face_img).unsqueeze(0).to(ensemble.device)
        img_f = ensemble.tf_flip(face_img).unsqueeze(0).to(ensemble.device)

        with torch.no_grad():
            # Model A TTA
            pred_a_o = ensemble.model_a(img_o)["pred_age"].item()
            pred_a_f = ensemble.model_a(img_f)["pred_age"].item()
            pred_a = 0.5 * pred_a_o + 0.5 * pred_a_f

            # Model B TTA
            pred_b_o = ensemble.model_b(img_o)["pred_age"].item()
            pred_b_f = ensemble.model_b(img_f)["pred_age"].item()
            pred_b = 0.5 * pred_b_o + 0.5 * pred_b_f

            # Ensemble Fusion
            ensemble_pred = 0.5 * pred_a + 0.5 * pred_b
            disagreement = abs(pred_a - pred_b)

        # Scale calibration
        ensemble_pred = ensemble_pred * 0.945
        final_age = round(float(ensemble_pred), 2)

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
            "model_a_dex": round(float(pred_a), 2),
            "model_b_hybrid": round(float(pred_b), 2),
            "disagreement": round(float(disagreement), 2),
            "bounds": {
                "pm_3": f"{max(1.0, round(final_age - 3.0, 1))} – {round(final_age + 3.0, 1)} yrs",
                "pm_5": f"{max(1.0, round(final_age - 5.0, 1))} – {round(final_age + 5.0, 1)} yrs",
                "pm_7": f"{max(1.0, round(final_age - 7.0, 1))} – {round(final_age + 7.0, 1)} yrs",
            },
            "latency_ms": latency_ms,
            "device": str(ensemble.device)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Inference failed: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
