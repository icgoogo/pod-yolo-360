# Pre-packaged with Python 3.10 + PyTorch + CUDA 12.1
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Install minimal OS dependencies for OpenCV & FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY *.pt .

VOLUME ["/app/output"]

CMD ["python", "app.py"]