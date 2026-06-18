# OCR (boarddetection) — Xiangqi board recognition, YOLO on NVIDIA GPU.
#
# Base image already bundles torch 2.5.1 + torchvision 0.20.1 built for
# CUDA 12.1 (cu121) — exactly what requirements-gpu.txt pins. So we do NOT
# reinstall torch here (saves the ~2.5GB cu121 download and removes any
# chance of pip grabbing the CPU wheel by mistake).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# OpenCV (cv2) links libGL + glib even in headless mode; the slim pytorch
# base doesn't ship them, so `import cv2` would crash at startup without these.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# App deps only (torch/torchvision come from the base). ultralytics declares
# opencv-python (the GUI build) as a dep, so we swap it for the headless build
# afterwards — the image then carries exactly one cv2 and no GUI baggage,
# matching requirements-gpu.txt.
RUN pip install --no-cache-dir \
        fastapi==0.115.6 \
        "uvicorn[standard]==0.32.1" \
        python-multipart==0.0.20 \
        ultralytics==8.3.55 \
        numpy==2.1.3 \
    && (pip uninstall -y opencv-python opencv-python-headless || true) \
    && pip install --no-cache-dir opencv-python-headless==4.10.0.84

# Package source. models/ is excluded by .dockerignore and bind-mounted
# read-only at runtime, so the ~28MB .pt assets never bake into the image
# (they must not end up in a pushed image — see docs/ocr-gpu-deploy.md).
COPY . /app/boarddetection/

# Pre-create the mount point so the bind target always exists.
RUN mkdir -p /app/boarddetection/models

EXPOSE 8001

# Run as a package from /app so server.py's `from .pipeline import ...` resolves.
CMD ["uvicorn", "boarddetection.server:app", "--host", "0.0.0.0", "--port", "8001"]
