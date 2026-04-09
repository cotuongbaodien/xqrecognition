"""
Training script for unified item detection model.
Detects 18 classes: 14 pieces + 4 board landmarks (board-conner, palace-bottom,
palace-center, palace-conner) in a single forward pass.
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from config.settings import MODELS_DIR


def train_items(
    data_yaml: str = "data/items/data.yaml",
    epochs: int = 200,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "cuda",
    pretrained: str = "yolov8s.pt",
    seed: int = 42,
    name: str = "items",
):
    data_path = Path(data_yaml)

    print("=" * 60)
    print("Item Detection Model Training (pieces + landmarks)")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | ImgSize: {img_size}")
    print(f"Pretrained: {pretrained} | Seed: {seed}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(pretrained)

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        patience=30,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "items"),
        name=name,
        exist_ok=False,
        seed=seed,
        # Augmentation - aggressive since dataset is small (153 images)
        flipud=0.0,            # Don't flip vertically (pieces have orientation)
        fliplr=0.5,            # Horizontal flip ok (board is symmetric)
        degrees=10.0,          # Mild rotation
        perspective=0.0005,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4,
        scale=0.4,
        translate=0.1,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    out = MODELS_DIR / "items.pt"
    backup = MODELS_DIR / f"items_{name}.pt"
    if best.exists():
        shutil.copy(best, out)
        shutil.copy(best, backup)
        print(f"\nBest model saved: {out}")
        print(f"Backup saved: {backup}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/items/data.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", default="yolov8s.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="items")
    args = parser.parse_args()

    train_items(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        pretrained=args.pretrained,
        seed=args.seed,
        name=args.name,
    )
