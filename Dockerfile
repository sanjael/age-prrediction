# =============================================================================
# Production Multi-Stage Dockerfile for AI Facial Age Estimation Backend
# =============================================================================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and backend files
COPY models.py config.py predict_dual_ensemble.py /app/
COPY backend/ /app/backend/
COPY outputs/ /app/outputs/

EXPOSE 8000

# Start production uvicorn server
CMD ["sh", "-c", "uvicorn backend.server_dual:app --host 0.0.0.0 --port ${PORT:-8000}"]
