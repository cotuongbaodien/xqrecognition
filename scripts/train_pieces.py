"""
Training script for the pieces detection model.
Trains a YOLOv8 model to detect and classify the 14 types of Xiangqi pieces.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from config.settings import (
    PIECES_DATA,
    PIECES_DET_MODEL,
    MODELS_DIR,
    TRAIN_CONFIG,
)


def find_data_yaml(data_dir: Path) -> Path:
    """Find the data.yaml file in the dataset directory."""
    # Check common locations
    candidates = [
        data_dir / "data.yaml",
        data_dir / "dataset.yaml",
    ]

    # Also search subdirectories (in case zip extracts to a subdirectory)
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            candidates.append(subdir / "data.yaml")
            candidates.append(subdir / "dataset.yaml")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find data.yaml in {data_dir}")


def train_pieces_model(
    data_yaml: str = None,
    epochs: int = None,
    batch_size: int = None,
    img_size: int = None,
    device: str = None,
    resume: bool = False,
    pretrained: str = "yolov8n.pt",
):
    """
    Train the pieces detection model.

    Args:
        data_yaml: Path to data.yaml file.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        img_size: Input image size.
        device: Device to train on ('cpu', 'cuda', 'mps', or 'auto').
        resume: Whether to resume from last checkpoint.
        pretrained: Pretrained model to use as base.
    """
    # Use config defaults if not specified
    epochs = epochs or TRAIN_CONFIG["epochs"]
    batch_size = batch_size or TRAIN_CONFIG["batch_size"]
    img_size = img_size or TRAIN_CONFIG["img_size"]
    device = device or TRAIN_CONFIG["device"]

    # Find data.yaml
    if data_yaml:
        data_path = Path(data_yaml)
    else:
        data_path = find_data_yaml(PIECES_DATA)

    print("=" * 60)
    print("Pieces Detection Model Training")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print(f"Resume: {resume}")
    print("=" * 60)

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize model
    if resume and PIECES_DET_MODEL.exists():
        print(f"Resuming from {PIECES_DET_MODEL}")
        model = YOLO(str(PIECES_DET_MODEL))
    else:
        print(f"Starting from pretrained: {pretrained}")
        model = YOLO(pretrained)

    # Train with augmentation for robustness across domains
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device if device != "auto" else None,
        patience=TRAIN_CONFIG["patience"],
        save=True,
        project=str(PROJECT_ROOT / "runs" / "pieces_det"),
        name="train",
        exist_ok=False,
        # Augmentation for domain robustness
        flipud=0.0,          # Không lật dọc (quân cờ có hướng)
        fliplr=0.5,          # Lật ngang
        degrees=15.0,         # Xoay ±15°
        perspective=0.0005,   # Perspective transform nhẹ
        hsv_h=0.02,          # Biến đổi hue
        hsv_s=0.3,           # Biến đổi saturation (ánh sáng khác nhau)
        hsv_v=0.3,           # Biến đổi brightness
        scale=0.3,           # Scale variation
        mosaic=1.0,          # Mosaic augmentation
    )

    # Copy best model to models directory
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        import shutil
        shutil.copy(best_model, PIECES_DET_MODEL)
        print(f"\nBest model saved to: {PIECES_DET_MODEL}")

    print("\nTraining completed!")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train pieces detection model")
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--img-size", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device (cpu/cuda/mps/auto)")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--pretrained", type=str, default="yolov8s.pt",
                        help="Pretrained model (yolov8s.pt recommended for accuracy)")

    args = parser.parse_args()

    train_pieces_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        resume=args.resume,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
