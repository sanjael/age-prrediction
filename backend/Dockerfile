# Use official lightweight Python with PyTorch CPU/CUDA support
FROM python:3.10-slim

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install essential system packages for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn opencv-python-headless timm

# Copy backend files and model code
COPY models.py config.py predict_dual_ensemble.py /app/
COPY backend/server_dual.py /app/server_dual.py
COPY outputs/ /app/outputs/

# Hugging Face Spaces runs on port 7860 with non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

# Start FastAPI server on port 7860
CMD ["uvicorn", "server_dual:app", "--host", "0.0.0.0", "--port", "7860"]
