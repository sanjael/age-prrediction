"""
views.py
Django REST Framework API Views for AgeVision Dual Ensemble.
Handles:
  1. POST /api/predict/    -> Local PyTorch Dual Ensemble (Model A DEX + Model B Hybrid + TTA)
  2. POST /api/hf-predict/ -> Remote Hugging Face Inference API
  3. GET  /api/health/     -> Service health & CUDA status
  4. GET  /api/metrics/    -> Tournament research benchmarks
"""

import io
import base64
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view

from .inference import DualEnsemblePredictor
from .hf_client import HuggingFaceInferenceClient

predictor = DualEnsemblePredictor.get_instance()
hf_client = HuggingFaceInferenceClient()


@api_view(['GET'])
def health_check(request):
    """Health check endpoint to verify backend status."""
    return Response({
        "status": "online",
        "service": "AGE-X AI Facial Age Prediction API",
        "version": "v3.3.0-django",
        "champion_model": "Dual Ensemble (EfficientNetV2-S DEX + Hybrid Head)",
        "final_mae": 4.18,
        "acc_plus_minus_7": 82.0,
        "acc_plus_minus_5": 67.0,
        "device": str(predictor.device),
        "is_model_loaded": predictor.is_loaded
    })


@api_view(['POST'])
def predict_face(request):
    """
    Primary Facial Age Prediction Endpoint.
    Accepts:
      - JSON: {"image": "data:image/jpeg;base64,...", "tta": true}
      - Form Multipart: file=<image file>, tta=true
    Returns:
      {
        "predicted_age": 25.4,
        "age_group": "Young Adult (20–35 yrs)",
        "model_a_dex": 25.1,
        "model_b_hybrid": 25.7,
        "disagreement": 0.6,
        "bounds": { "pm_3": "22.4 – 28.4 yrs", ... },
        "latency_ms": 28.5
      }
    """
    tta = request.data.get('tta', True)
    if isinstance(tta, str):
        tta = tta.lower() in ('true', '1')

    image = None

    # 1. Check for file upload
    if 'file' in request.FILES:
        try:
            uploaded_file = request.FILES['file']
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            return Response({"error": f"Failed to parse uploaded file: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

    # 2. Check for base64 string
    elif 'image' in request.data:
        try:
            base64_str = request.data['image']
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return Response({"error": f"Failed to decode base64 image data: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

    if image is None:
        return Response({"error": "No image provided. Please send 'image' as base64 or 'file' in multipart form."}, status=status.HTTP_400_BAD_REQUEST)

    # Run Dual Ensemble Inference
    try:
        result = predictor.predict_image(image, use_tta=tta)
        return Response(result, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": f"Inference pipeline failure: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def hf_predict(request):
    """Remote inference proxy connecting to Hugging Face Model Endpoint."""
    image = None
    if 'file' in request.FILES:
        image = Image.open(request.FILES['file']).convert("RGB")
    elif 'image' in request.data:
        base64_str = request.data['image']
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")

    if image is None:
        return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

    result = hf_client.predict_image(image)
    return Response(result)


@api_view(['GET'])
def get_metrics(request):
    """Returns official research benchmarks for frontend dashboard consumption."""
    return Response({
        "final_ensemble_mae": 4.18,
        "master_corpus_count": 276280,
        "accuracy_pm_7": 82.0,
        "accuracy_pm_5": 67.0,
        "tournament_models": [
            {"id": "EXP-01", "name": "ResNet-50 (L1)", "mae": 7.62},
            {"id": "EXP-DF25K", "name": "DeepFake-25K Fine-Tuned (8 Epochs)", "mae": 7.20},
            {"id": "EXP-15", "name": "MobileNetV3", "mae": 6.45},
            {"id": "EXP-20", "name": "ResNet-50 + Huber", "mae": 5.11},
            {"id": "EXP-23", "name": "EffNetV2-S (Hybrid Head)", "mae": 4.67},
            {"id": "EXP-25", "name": "EffNetV2-S (DEX Head)", "mae": 4.64},
            {"id": "DUAL-ENS", "name": "Dual Ensemble (DEX + Hybrid + TTA)", "mae": 4.18, "is_champion": True}
        ]
    })
