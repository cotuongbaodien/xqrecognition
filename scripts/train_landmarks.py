"""
Training script for board landmark detection model.
Detects: corners (4), palace_red, palace_black, river.
Used for accurate grid construction and mirror detection.
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from config.settings import MODELS_DIR


def train_landmarks(
    data_yaml: str = None,
    epochs: int = 150,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "auto",
    pretrained: str = "yolov8s.pt",
):
    data_path = Path(data_yaml or "data/landmarks/data.yaml")

    print("=" * 60)
    print("Landmark Detection Model Training")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Classes: corner, palace_red, palace_black, river")
    print(f"Epochs: {epochs}, Batch: {batch_size}, ImgSize: {img_size}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(pretrained)

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device if device != "auto" else None,
        patience=20,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "landmarks"),
        name="train",
        exist_ok=True,
        # Augmentation
        flipud=0.0,
        fliplr=0.5,
        degrees=10.0,
        perspective=0.0003,
        hsv_h=0.02,
        hsv_s=0.3,
        hsv_v=0.3,
        scale=0.3,
        mosaic=1.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    out = MODELS_DIR / "landmarks.pt"
    if best.exists():
        shutil.copy(best, out)
        print(f"\nModel saved: {out}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/landmarks/data.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pretrained", type=str, default="yolov8s.pt")
    args = parser.parse_args()

    train_landmarks(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        pretrained=args.pretrained,
    )
