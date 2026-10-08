FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies (CPU torch for HF Spaces)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi==0.115.5 uvicorn[standard]==0.32.1 python-multipart==0.0.12 pydantic==2.10.3
RUN pip install --no-cache-dir ultralytics==8.3.55 opencv-python-headless==4.10.0.84 numpy==1.26.4 matplotlib==3.9.3 PyYAML==6.0.2 requests==2.32.3

# Copy application code
COPY app.py .
COPY model_b_engine.py .
COPY dataset_manager.py .
COPY simulation_engine.py .
COPY static/ ./static/
COPY models/Model_B_v2_best.pt ./models/Model_B_v2_best.pt
COPY models/class_names.txt ./models/class_names.txt
COPY models/MODEL_INFO.txt ./models/MODEL_INFO.txt

# Create necessary directories
RUN mkdir -p debug_samples data/model_b_webcam_v2/raw

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Start server on port 7860
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
