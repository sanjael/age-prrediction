# AI-Powered Facial Age Prediction 🎯

Real-time facial age estimation using a **Dual Ensemble Architecture** combining:
1. **Model A:** EfficientNetV2-S with DEX Head (100-way Softmax Expected Value)
2. **Model B:** EfficientNetV2-S with Hybrid Head (Regression + Categorical Anchoring)
3. **2-View Mirror Test-Time Augmentation (TTA)** (Original + Horizontal Flip)
4. **Final Accuracy:** **4.18 Years MAE**, **82% within ±7 Years**, **67% within ±5 Years**.

---

## 📁 Folder Structure

```text
DATA SET/
├── backend/                  # API backend (FastAPI & Django REST Framework)
│   ├── server_dual.py        # High-performance FastAPI server with CUDA & OpenCV Face Crop
│   ├── manage.py             # Django management script
│   ├── age_vision_backend/   # Django project settings & CORS configuration
│   └── api/                  # Django REST API app (inference, Hugging Face client, views)
│
├── frontend/                 # Web Application (Research Dashboard & Live Studio)
│   ├── index.html            # Research & Experimentation Dashboard (Tournament, 5-Stage Flowchart)
│   ├── demo.html             # Live Studio (Live Webcam Auto-Scan & Image Upload)
│   ├── css/style.css         # Styling stylesheet
│   └── js/                   # Frontend logic (app.js, charts.js, data.js, demo.js)
│
├── outputs/                  # Model Checkpoints & Training Logs
│   ├── exp25_effnetv2s_dex_expected_age/best_model.pt   # Model A (DEX Head)
│   └── exp23_effnetv2s_utkface_supplement/best_model.pt # Model B (Hybrid Head)
│
├── web_app/                  # Additional web assets & static sample images
│   └── static/samples/       # Test demographic sample images
│
├── predict_dual_ensemble.py  # Master Dual Ensemble CLI script (Single image & full evaluation)
├── predict_single_image.py   # Single model prediction script
├── test_model.py             # Evaluation & testing scripts
├── models.py                 # PyTorch model architecture definitions (AgeModel)
├── config.py                 # Hyperparameters & dataset paths
├── dataset.py                # PyTorch dataset & augmentation loaders
├── requirements.txt          # Core dependencies
└── API_DESIGN_ARCHITECTURE.md# Complete API and deployment documentation
```

---

## 🚀 How to Run

### 1. Single Image Age Prediction (CLI)

Run inference on any individual face image:

```powershell
python predict_dual_ensemble.py --image "data_processed/imdb_faces/imdb_000022_age_10.jpg"
```

Or test with sample images:

```powershell
python predict_dual_ensemble.py --image "web_app/static/samples/sample_young_woman.png"
```

**Example Output:**
```text
=================================================================
 GLOBAL DUAL ENSEMBLE PREDICTION RESULT
=================================================================
 Image Path          : data_processed/imdb_faces/imdb_000022_age_10.jpg
 Predicted Final Age : 12.88 years old
  - Model A (DEX)    : 14.99 yrs
  - Model B (Hybrid) : 12.27 yrs
  - Disagreement     : 2.72 yrs
 Age Cohort Category : Teenager (13-19)
=================================================================
```

---

### 2. Whole Validation / Full Test Evaluation

To run complete evaluation across the master test manifest:

```powershell
python predict_dual_ensemble.py --evaluate
```

Or evaluate a specific test split:

```powershell
python test_model.py --manifest "manifest_p2_320_plus_utkface.csv" --split test
```

---

### 3. Run Backend API Server (Port 8000)

Start the live CUDA inference API:

```powershell
python backend/server_dual.py
```

* Backend will start at: `http://localhost:8000`
* Health Check Endpoint: `http://localhost:8000/api/health`
* Prediction Endpoint: `POST http://localhost:8000/api/predict`

---

### 4. Run Frontend Dashboard & Live Studio (Port 8080)

In a separate terminal, serve the frontend:

```powershell
cd frontend
python -m http.server 8080
```

Open in your browser:
* **Research & Experimentation Dashboard:** `http://localhost:8080/index.html`
* **Live Biometric Age Studio:** `http://localhost:8080/demo.html`

---

## 🌐 Hugging Face Deployment

* **Model Repository:** [`Sanjayramdata/celingfan`](https://huggingface.co/Sanjayramdata/celingfan)
* Remote inference integration is configured in `backend/api/hf_client.py`.
