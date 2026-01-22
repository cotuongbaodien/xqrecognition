"""
Training script for the board segmentation model.
Trains a YOLOv8-seg model to detect intersection points on the board.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from config.settings import (
    BOARD_SEG_DATA,
    BOARD_SEG_MODEL,
    MODELS_DIR,
    BOARD_SEG_TRAIN_CONFIG,
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


def train_board_model(
    data_yaml: str = None,
    epochs: int = None,
    batch_size: int = None,
    img_size: int = None,
    device: str = None,
    resume: bool = False,
    pretrained: str = "yolov8n-seg.pt",
):
    """
    Train the board segmentation model.

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
    epochs = epochs or BOARD_SEG_TRAIN_CONFIG["epochs"]
    batch_size = batch_size or BOARD_SEG_TRAIN_CONFIG["batch_size"]
    img_size = img_size or BOARD_SEG_TRAIN_CONFIG["img_size"]
    device = device or BOARD_SEG_TRAIN_CONFIG["device"]

    # Find data.yaml
    if data_yaml:
        data_path = Path(data_yaml)
    else:
        data_path = find_data_yaml(BOARD_SEG_DATA)

    print("=" * 60)
    print("Board Segmentation Model Training")
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
    if resume and BOARD_SEG_MODEL.exists():
        print(f"Resuming from {BOARD_SEG_MODEL}")
        model = YOLO(str(BOARD_SEG_MODEL))
    else:
        print(f"Starting from pretrained: {pretrained}")
        model = YOLO(pretrained)

    # Train with augmentation settings matching Roboflow preprocessing
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device if device != "auto" else None,
        patience=BOARD_SEG_TRAIN_CONFIG["patience"],
        save=True,
        project=str(PROJECT_ROOT / "runs" / "board_seg"),
        name="train",
        exist_ok=True,
        # Augmentation settings matching Roboflow
        flipud=0.5,         # Vertical flip probability
        fliplr=0.5,         # Horizontal flip probability
        degrees=15.0,       # Rotation ±15°
        shear=13.0,         # Shear ±13°
        hsv_h=0.015,        # HSV-Hue augmentation
        hsv_s=0.15,         # HSV-Saturation (brightness)
        hsv_v=0.1,          # HSV-Value (exposure)
        mosaic=1.0,         # Mosaic augmentation
    )

    # Copy best model to models directory
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    if best_model.exists():
        import shutil
        shutil.copy(best_model, BOARD_SEG_MODEL)
        print(f"\nBest model saved to: {BOARD_SEG_MODEL}")

    print("\nTraining completed!")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train board segmentation model")
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--img-size", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device (cpu/cuda/mps/auto)")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--pretrained", type=str, default="yolov8n-seg.pt",
                        help="Pretrained model")

    args = parser.parse_args()

    train_board_model(
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
