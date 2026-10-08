FROM python:3.11-slim

# System dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch for lightning-fast build and small image footprint
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    fastapi==0.115.5 \
    uvicorn[standard]==0.32.1 \
    python-multipart==0.0.12 \
    pydantic==2.10.3 \
    ultralytics==8.3.55 \
    opencv-python-headless==4.10.0.84 \
    numpy==1.26.4 \
    matplotlib==3.9.3 \
    PyYAML==6.0.2 \
    requests==2.32.3

# Hugging Face Spaces standard non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy repo contents with user ownership
COPY --chown=user:user . $HOME/app

# Ensure writable directories exist
RUN mkdir -p debug_samples data/model_b_webcam_v2/raw

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
