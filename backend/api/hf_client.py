"""
hf_client.py
Hugging Face Inference API Client for Remote Model Deployment.
Handles communication with Hugging Face Inference Endpoints or Spaces.
"""

import os
import io
import time
import requests
from PIL import Image

class HuggingFaceInferenceClient:
    def __init__(self, api_key: str = None, model_repo: str = None):
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY", "")
        self.model_repo = model_repo or os.environ.get("HUGGINGFACE_MODEL_REPO", "Sanjayramdata/celingfan")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_repo}"

    def predict_image(self, pil_image: Image.Image) -> dict:
        """Sends image bytes to Hugging Face Inference API."""
        start_time = time.time()
        
        # Convert PIL to JPEG bytes
        buffered = io.BytesIO()
        pil_image.convert("RGB").save(buffered, format="JPEG", quality=90)
        image_bytes = buffered.getvalue()

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(self.api_url, headers=headers, data=image_bytes, timeout=15)
            if response.status_code == 200:
                data = response.json()
                latency_ms = round((time.time() - start_time) * 1000, 1)
                return {
                    "source": "hugging_face_endpoint",
                    "status": "success",
                    "data": data,
                    "latency_ms": latency_ms
                }
            else:
                return {
                    "source": "hugging_face_endpoint",
                    "status": "error",
                    "error": response.text,
                    "status_code": response.status_code
                }
        except Exception as e:
            return {
                "source": "hugging_face_endpoint",
                "status": "fallback",
                "message": f"HF Connection exception: {str(e)}"
            }
