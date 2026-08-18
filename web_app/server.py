import os
import io
import time
import uuid
import base64
import numpy as np
import cv2
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import AgeModel
from predict_single_image import SingleImageAgePredictor
from database import init_db, log_scan, get_recent_logs

app = FastAPI(title="Cognizant AgeVision AI Enterprise Suite", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global Predictor Instance
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    init_db()
    print("[*] Initializing Tri-Model Super-Ensemble for Web API...")
    try:
        predictor = SingleImageAgePredictor()
        print("[+] AI Engine loaded and ready for live web requests!")
    except Exception as e:
        print(f"[-] AI Model Load Error: {e}")

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Cognizant AgeVision AI Suite is running. Please add index.html to web_app/static/</h1>"

@app.get("/api/review-metrics")
def get_review_metrics():
    """Returns official tournament benchmark data for Dashboard 1."""
    return {
        "project_name": "Cognizant AgeVision AI - Enterprise Demographics Suite",
        "total_dataset_images": 233856,
        "validation_images": 16258,
        "champion_metrics": {
            "overall_mae": 4.31,
            "overall_rmse": 6.23,
            "accuracy_pm_3": 48.76,
            "accuracy_pm_5": 69.61,
            "accuracy_pm_7": 82.40,
            "accuracy_pm_10": 91.68,
            "core_population_mae": 3.93,
            "children_mae": 2.63
        },
        "tournament_leaderboard": [
            {
                "id": "EXP-20",
                "name": "Baseline (ResNet50 + Huber)",
                "backbone": "ResNet-50",
                "head": "Linear Regression",
                "data": "1.49 Lakh images",
                "mae": 5.11,
                "acc_7": 75.4,
                "badge": "Baseline"
            },
            {
                "id": "EXP-23",
                "name": "Leader (EffNetV2-S + Hybrid)",
                "backbone": "EfficientNetV2-S",
                "head": "Hybrid Dual Head",
                "data": "1.70 Lakh images (UTKFace)",
                "mae": 4.67,
                "acc_7": 78.1,
                "badge": "Model 2"
            },
            {
                "id": "EXP-25",
                "name": "Champion (EffNetV2-S + DEX)",
                "backbone": "EfficientNetV2-S",
                "head": "DEX 100-Way Softmax",
                "data": "1.70 Lakh images (UTKFace)",
                "mae": 4.64,
                "acc_7": 78.2,
                "badge": "Model 1"
            },
            {
                "id": "EXP-27",
                "name": "Vision Transformer (ConvNeXt-Tiny)",
                "backbone": "ConvNeXt-Tiny (7x7 Inv)",
                "head": "DEX 100-Way Softmax",
                "data": "1.70 Lakh images (UTKFace)",
                "mae": 4.63,
                "acc_7": 78.5,
                "badge": "Model 3"
            },
            {
                "id": "TRI-SUPER",
                "name": "Tri-Model Super-Ensemble + TTA",
                "backbone": "EffNet + ConvNeXt Fusion",
                "head": "Tri-Fusion + Spline Cal",
                "data": "2.33 Lakh Master Manifest",
                "mae": 4.31,
                "acc_7": 82.4,
                "badge": "Grand Champion"
            }
        ],
        "age_group_breakdown": [
            {"group": "01-12 (Children)", "mae": 2.63, "acc_5": 85.93, "acc_7": 88.32, "acc_10": 93.11, "grade": "Exceptional"},
            {"group": "13-19 (Teens)", "mae": 5.31, "acc_5": 59.92, "acc_7": 73.45, "acc_10": 86.47, "grade": "Good"},
            {"group": "20-30 (Young Adults)", "mae": 3.74, "acc_5": 76.13, "acc_7": 86.80, "acc_10": 94.20, "grade": "Exceptional"},
            {"group": "31-45 (Adults)", "mae": 3.98, "acc_5": 71.21, "acc_7": 85.29, "acc_10": 93.95, "grade": "High Accuracy"},
            {"group": "46-60 (Middle Age)", "mae": 5.13, "acc_5": 60.94, "acc_7": 75.19, "acc_10": 87.41, "grade": "Good"},
            {"group": "61-75 (Seniors)", "mae": 5.97, "acc_5": 56.86, "acc_7": 71.35, "acc_10": 84.06, "grade": "Moderate"},
            {"group": "76-100 (Elderly)", "mae": 8.59, "acc_5": 46.85, "acc_7": 58.27, "acc_10": 70.87, "grade": "Sparse Data"}
        ],
        "epoch_curves": {
            "epochs": [1, 2, 3, 4, 5],
            "train_mae": [7.84, 5.30, 4.61, 3.99, 3.62],
            "val_mae": [5.73, 4.90, 4.82, 4.69, 4.64],
            "val_acc_5": [54.58, 63.30, 64.44, 65.92, 66.69]
        }
    }

@app.post("/api/predict")
async def predict_face(file: UploadFile = File(...), use_case: str = Form("KYC_Verification")):
    global predictor
    if predictor is None:
        predictor = SingleImageAgePredictor()
        
    start_time = time.time()
    contents = await file.read()
    
    # Open image with Exif orientation auto-correction
    from PIL import ImageOps
    try:
        raw_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = ImageOps.exif_transpose(raw_image)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "ERROR_INVALID_IMAGE", "message": "The uploaded file is not a valid image format."}
        )
        
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 1. Multi-Angle Strict Human Face Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Try 0 deg, 90 deg, 270 deg for sideways documents like PAN cards
    detected_faces = []
    best_rotation = 0
    
    for rot in [0, 90, 270, 180]:
        if rot == 0:
            test_gray = gray
        elif rot == 90:
            test_gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 270:
            test_gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            test_gray = cv2.rotate(gray, cv2.ROTATE_180)
            
        faces = face_cascade.detectMultiScale(test_gray, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30))
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(test_gray, scaleFactor=1.08, minNeighbors=3, minSize=(30, 30))
            
        if len(faces) > 0:
            detected_faces = faces
            best_rotation = rot
            break
            
    # If no face is detected anywhere in the image
    if len(detected_faces) == 0:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ERROR_NO_FACE",
                "message": "⚠️ No Valid Human Face Detected! Please upload a clear front-facing photograph, selfie, or ID photo with a visible face."
            }
        )
        
    # Rotate working image if needed
    if best_rotation == 90:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        image = image.rotate(270, expand=True)
    elif best_rotation == 270:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = image.rotate(90, expand=True)
    elif best_rotation == 180:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
        image = image.rotate(180, expand=True)
        
    x, y, w, h = [int(v) for v in detected_faces[0]]
    bounding_box = {"x": x, "y": y, "width": w, "height": h}
    
    # Save temporary cropped face / aligned image for inference
    temp_path = f"temp_upload_{uuid.uuid4().hex[:8]}.png"
    image.save(temp_path)
    
    try:
        res = predictor.predict(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    latency = int((time.time() - start_time) * 1000)
    scan_id = f"CTS-{uuid.uuid4().hex[:8].upper()}"
    
    # Business Logic by Use Case
    client_insights = {}
    pred_age = res["final_predicted_age"]
    if use_case == "KYC_Verification":
        if pred_age < 18:
            client_insights = {"status": "FLAG_UNDERAGE", "action": "Block transaction / Request Parent Consent", "tier": "Minor (< 18 yrs)"}
        elif pred_age >= 60:
            client_insights = {"status": "SENIOR_PRIORITY", "action": "Route to Senior Citizen Fast-Track Lounge", "tier": "Senior Citizen (60+ yrs)"}
        else:
            client_insights = {"status": "VERIFIED_ADULT", "action": "Standard KYC Processing Approved", "tier": "Adult (18-59 yrs)"}
    elif use_case == "Retail_Smart_Ads":
        if pred_age < 13:
            client_insights = {"status": "TARGET_KIDS", "ad_theme": "Gaming, Toys & Kids Snacks Campaign", "banner": "Theme Park & Fun zone Promo"}
        elif pred_age <= 25:
            client_insights = {"status": "TARGET_GENZ", "ad_theme": "Trendy Fashion, Smartphones & Sneakers", "banner": "College Tech Pass Discount"}
        elif pred_age <= 50:
            client_insights = {"status": "TARGET_PROFESSIONAL", "ad_theme": "Automobiles, Home Loans & Executive Apparel", "banner": "Premium Platinum Card Upgrade"}
        else:
            client_insights = {"status": "TARGET_SENIOR", "ad_theme": "Healthcare, Wellness Retreats & Investment Bonds", "banner": "Senior Wellness Gold Plan"}
    else: # Healthcare_Triage
        if pred_age < 14:
            client_insights = {"status": "PEDIATRIC_TRIAGE", "department": "Pediatric Emergency & Child Wellness", "priority": "High (Children Protocol)"}
        elif pred_age >= 65:
            client_insights = {"status": "GERIATRIC_CARE", "department": "Geriatric Specialist & Cardiology Triage", "priority": "High (Senior Care Protocol)"}
        else:
            client_insights = {"status": "GENERAL_MEDICINE", "department": "General Outpatient Consultation", "priority": "Standard"}
            
    # Convert image to base64 for MySQL DB storage
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_b64_str = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

    # Log to MySQL DB
    log_scan(
        scan_id=scan_id,
        image_name=file.filename,
        image_base64=img_b64_str,
        predicted_age=pred_age,
        confidence_range=res["confidence_range"],
        age_category=res["age_category"],
        m_dex=res.get("dex_pred", 0.0),
        m_hyb=res.get("hybrid_pred", 0.0),
        m_cnx=res.get("convnext_pred", 0.0),
        client_use_case=use_case,
        latency_ms=latency
    )
    
    return {
        "scan_id": scan_id,
        "predicted_age": pred_age,
        "confidence_range": res["confidence_range"],
        "age_category": res["age_category"],
        "models_breakdown": {
            "model_1_dex": res.get("dex_pred", pred_age),
            "model_2_hybrid": res.get("hybrid_pred", pred_age),
            "model_3_convnext": res.get("convnext_pred", pred_age),
            "final_ensemble": pred_age
        },
        "bounding_box": bounding_box,
        "client_insights": client_insights,
        "latency_ms": latency
    }

@app.get("/api/audit-logs")
def get_audit_logs():
    return get_recent_logs(limit=20)

@app.get("/api/export-csv")
def export_csv():
    logs = get_recent_logs(limit=1000)
    import pandas as pd
    df = pd.DataFrame(logs)
    csv_path = "cts_agevision_audit_export.csv"
    df.to_csv(csv_path, index=False)
    return FileResponse(csv_path, filename="Cognizant_AgeVision_Audit_Report.csv", media_type="text/csv")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
