# AGE-X AI Facial Age Prediction | Full Stack & API Architecture Design

This document details the end-to-end architecture connecting the **Research & Experimentation Dashboard**, the **Live Biometric Studio (`/demo`)**, the **Django REST API Backend**, and the **Hugging Face Model Deployment Pipeline**.

---

## 🏛️ 1. Complete System Architecture

```
+---------------------------------------------------------------------------------------+
|                                    CLIENT LAYER                                       |
|                                                                                       |
|  [ Research Dashboard: index.html ]        [ Live Biometric Studio: demo.html ]      |
|  • Tournament Benchmarks (4.18y MAE)       • Live Webcamera Auto-Scan                 |
|  • 5-Stage Architecture Flowchart          • Image File Drag & Drop                   |
|  • Convergence Curves & Cohorts            • Real-time Confidence Score & Cohorts     |
+------------------------------------------┬--------------------------------------------+
                                           │ HTTP REST / JSON (Base64 / Multipart)
                                           ▼
+---------------------------------------------------------------------------------------+
|                                 DJANGO REST BACKEND                                   |
|                                  (Port 8000)                                          |
|                                                                                       |
|   • urls.py               -> Routing (/api/predict/, /api/hf-predict/, /api/health/)  |
|   • views.py              -> Request parsing, Base64 decode, TTA toggle, Serializers  |
|   • inference.py          -> Local PyTorch Dual Ensemble (CUDA/CPU)                   |
|   • hf_client.py          -> Hugging Face Inference API Proxy                         |
|   • settings.py           -> CORS headers, REST framework, Static configuration      |
+------------------------------------------┬--------------------------------------------+
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
+---------------------------------------+     +---------------------------------------+
|         LOCAL INFERENCE ENGINE        |     |     HUGGING FACE DEPLOYMENT API       |
|                                       |     |                                       |
|  1. Preprocessing (320x320 + Flip)   |     |  • Model Repo:                        |
|  • Model Repo:                        |     |    sanjaelraja/facial-age-prediction  |
|    Sanjayramdata/celingfan            |     |  • Inference Endpoint / Spaces API    |
|  3. Model B: EffNetV2-S (Hybrid Head) |     |  • Authentication Bearer Token        |
|  4. Dual Ensemble Fusion              |     |                                       |
|  5. Output: Age (4.18y) + Confidence  |     |                                       |
+---------------------------------------+     +---------------------------------------+
```

---

## 📁 2. File-by-File Directory Mapping

### 🌐 Frontend Directory (`frontend/`)
| File | Role & Responsibilities |
| :--- | :--- |
| `frontend/index.html` | **Research & Experimentation Dashboard** containing KPI cards (4.18 MAE, 82% ±7y, 67% ±5y), Model Tournament (including DeepFake 25k 7.20 MAE), 5-Stage Dual Architecture Flowchart, Convergence & Cohort Charts, and the **"START LIVE DEMO →"** CTA. |
| `frontend/demo.html` | **Live Biometric Studio** featuring Live Webcamera auto-scan, Image Upload dropzone, Quick demographic sample presets, real-time Confidence Gauge, Model A & B breakdown cards, and Session Audit Telemetry. |
| `frontend/css/style.css` | **Strict 50/30/20 Design System stylesheet** (#0B0D10 canvas, #15181D/#1B1F25 cards, #F5B942 amber accent, Inter typography, 12-column grid, and flowchart styles). |
| `frontend/js/data.js` | **Centralized Data Repository** maintaining exact metrics, cohort numbers, experiment logs, and architecture rationales. |
| `frontend/js/charts.js` | **Research-grade Chart.js visualizations** (Tournament bar chart, 1–100 continuous curve, 5-epoch dual-axis convergence curve, and cohort performance chart). |
| `frontend/js/app.js` | **Dashboard Controller** mounting components, scroll spy navigation, and dynamic renderers. |
| `frontend/js/demo.js` | **Live Studio Controller** managing `navigator.mediaDevices.getUserMedia` webcam stream, auto-scan timers, file upload decoding, and sending requests to the Django REST backend. |

---

### ⚙️ Django Backend Directory (`backend/`)
| File | Role & Responsibilities |
| :--- | :--- |
| `backend/manage.py` | Django command-line execution entry point. |
| `backend/requirements.txt` | Python package dependencies (`django`, `djangorestframework`, `django-cors-headers`, `torch`, `torchvision`, `pillow`, `requests`). |
| `backend/age_vision_backend/settings.py` | Django project settings configured with CORS headers (`CORS_ALLOW_ALL_ORIGINS = True`), REST Framework, and Hugging Face API keys. |
| `backend/age_vision_backend/urls.py` | Top-level URL dispatcher routing `/api/` traffic to the API app. |
| `backend/api/apps.py` | Application registration for the API module. |
| `backend/api/urls.py` | API routes dispatching `/api/health/`, `/api/predict/`, `/api/hf-predict/`, and `/api/metrics/`. |
| `backend/api/views.py` | REST API view controllers handling image parsing, validation, calling the Dual Ensemble engine, and returning formatted JSON responses. |
| `backend/api/inference.py` | **Dual Ensemble PyTorch inference engine** loading Model A (DEX Head from `outputs/exp25_...`) and Model B (Hybrid Head from `outputs/exp23_...`) with 2-View Mirror TTA. |
| `backend/api/hf_client.py` | **Hugging Face Client** communicating with remote Hugging Face Inference Endpoints or Spaces. |

---

## 📡 3. REST API Endpoint Specifications

### 🔹 Endpoint 1: Facial Age Prediction
* **URL:** `/api/predict/`
* **Method:** `POST`
* **Content-Type:** `application/json` or `multipart/form-data`
* **Request Payload (JSON):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "tta": true
}
```
* **Response Payload (`200 OK`):**
```json
{
  "predicted_age": 25.4,
  "age_group": "Young Adult (20–35 yrs)",
  "cohort_code": "YADULT_20_35",
  "model_a_dex": 25.1,
  "model_b_hybrid": 25.7,
  "disagreement": 0.6,
  "tta_applied": true,
  "bounds": {
    "pm_3": "22.4 – 28.4 yrs",
    "pm_5": "20.4 – 30.4 yrs",
    "pm_7": "18.4 – 32.4 yrs"
  },
  "latency_ms": 28.5,
  "device": "cuda",
  "timestamp": "2026-08-19T01:30:00Z"
}
```

---

### 🔹 Endpoint 2: Hugging Face Remote Proxy
* **URL:** `/api/hf-predict/`
* **Method:** `POST`
* **Payload:** Image file or base64 string
* **Response:** Formatted age estimation from Hugging Face Inference API.

---

### 🔹 Endpoint 3: System Health Check
* **URL:** `/api/health/`
* **Method:** `GET`
* **Response Payload (`200 OK`):**
```json
{
  "status": "online",
  "service": "AGE-X AI Facial Age Prediction API",
  "version": "v3.3.0-django",
  "champion_model": "Dual Ensemble (EfficientNetV2-S DEX + Hybrid Head)",
  "final_mae": 4.18,
  "acc_plus_minus_7": 82.0,
  "acc_plus_minus_5": 67.0,
  "device": "cuda",
  "is_model_loaded": true
}
```

---

## 🚀 4. How to Run the Complete Stack

### Step 1: Start Django Backend (Terminal 1)
```powershell
cd "e:\CTS FINAL\DATA SET\backend"
python manage.py runserver 8000
```

### Step 2: Start Frontend Web Server (Terminal 2)
```powershell
cd "e:\CTS FINAL\DATA SET\frontend"
python -m http.server 8080
```

### Step 3: Access in Browser
1. Open **`http://localhost:8080/index.html`** for the **Research & Experimentation Dashboard**.
2. Click **"START LIVE DEMO →"** to launch **`http://localhost:8080/demo.html`** for live webcam scanning and image age prediction.
