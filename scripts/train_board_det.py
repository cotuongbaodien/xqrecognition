"""
Training script for the board detection model.
Trains a YOLOv8 detection model to detect the board bounding box.
Uses "xiangqi game.v6i.yolov8" dataset.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

BOARD_DET_DATA = PROJECT_ROOT / "data" / "xiangqi_game"
BOARD_DET_MODEL = PROJECT_ROOT / "models" / "board_det.pt"
MODELS_DIR = PROJECT_ROOT / "models"


def train_board_detection(
    data_yaml: str = None,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = None,
    resume: bool = False,
    pretrained: str = "yolov8n.pt",
):
    """
    Train the board detection model.

    Args:
        data_yaml: Path to data.yaml file.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        img_size: Input image size.
        device: Device to train on.
        resume: Whether to resume from last checkpoint.
        pretrained: Pretrained model to use as base.
    """
    # Find data.yaml
    if data_yaml:
        data_path = Path(data_yaml)
    else:
        data_path = BOARD_DET_DATA / "data.yaml"

    print("=" * 60)
    print("Board Detection Model Training")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print("=" * 60)

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize model
    if resume and BOARD_DET_MODEL.exists():
        print(f"Resuming from {BOARD_DET_MODEL}")
        model = YOLO(str(BOARD_DET_MODEL))
    else:
        print(f"Starting from pretrained: {pretrained}")
        model = YOLO(pretrained)

    # Train with augmentation settings matching user config:
    # - Flip Horizontal
    # - Hue ±10° (±10/360 ≈ 0.028)
    # - Brightness ±20%
    # - Blur up to 2.3px (not directly supported, using other augmentations)
    # - Noise up to 0.93% (not directly supported)
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device if device else None,
        patience=20,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "board_det"),
        name="train",
        exist_ok=True,
        # Augmentation settings
        flipud=0.0,       # No vertical flip
        fliplr=0.5,       # Horizontal flip
        degrees=0.0,      # No rotation
        hsv_h=0.028,      # Hue ±10° (10/360)
        hsv_s=0.1,        # Some saturation variation
        hsv_v=0.2,        # Brightness ±20%
        mosaic=1.0,       # Mosaic augmentation
        scale=0.3,        # Scale variation
    )

    # Copy best model to models directory
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        import shutil
        shutil.copy(best_model, BOARD_DET_MODEL)
        print(f"\nBest model saved to: {BOARD_DET_MODEL}")

    print("\nTraining completed!")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train board detection model")
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, help="Device (cpu/cuda/mps)")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt",
                        help="Pretrained model")

    args = parser.parse_args()

    train_board_detection(
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
