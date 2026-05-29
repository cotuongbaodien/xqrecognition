"""
Training CLI for Xiangqi Recognition System.
Provides unified interface for training board and pieces models.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_board(args):
    """Train the board segmentation model."""
    from scripts.train_board import train_board_model

    train_board_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        resume=args.resume,
        pretrained=args.pretrained or "yolo11n-seg.pt",
    )


def train_pieces(args):
    """Train the pieces detection model."""
    from scripts.train_pieces import train_pieces_model

    train_pieces_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        resume=args.resume,
        pretrained=args.pretrained or "yolo11s.pt",
    )


def train_all(args):
    """Train both models."""
    print("=" * 60)
    print("Training Board Segmentation Model")
    print("=" * 60)
    train_board(args)

    print("\n" + "=" * 60)
    print("Training Pieces Detection Model")
    print("=" * 60)
    train_pieces(args)

    print("\n" + "=" * 60)
    print("All training completed!")
    print("=" * 60)


def setup_data(args):
    """Setup datasets by extracting zip files."""
    from scripts.setup_data import setup_datasets
    setup_datasets()


def main():
    parser = argparse.ArgumentParser(
        description="Xiangqi Recognition System - Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup datasets
  python train.py setup

  # Train board segmentation model
  python train.py board --epochs 100

  # Train pieces detection model
  python train.py pieces --epochs 100

  # Train both models
  python train.py all --epochs 100

  # Resume training
  python train.py pieces --resume
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Training command")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup datasets")

    # Board training command
    board_parser = subparsers.add_parser("board", help="Train board segmentation model")
    board_parser.add_argument("--data", type=str, help="Path to data.yaml")
    board_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    board_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    board_parser.add_argument("--img-size", type=int, default=640, help="Image size")
    board_parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/mps)")
    board_parser.add_argument("--resume", action="store_true", help="Resume training")
    board_parser.add_argument("--pretrained", type=str, help="Pretrained model path")

    # Pieces training command
    pieces_parser = subparsers.add_parser("pieces", help="Train pieces detection model")
    pieces_parser.add_argument("--data", type=str, help="Path to data.yaml")
    pieces_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    pieces_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    pieces_parser.add_argument("--img-size", type=int, default=640, help="Image size")
    pieces_parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/mps)")
    pieces_parser.add_argument("--resume", action="store_true", help="Resume training")
    pieces_parser.add_argument("--pretrained", type=str, help="Pretrained model path")

    # All training command
    all_parser = subparsers.add_parser("all", help="Train both models")
    all_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    all_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    all_parser.add_argument("--img-size", type=int, default=640, help="Image size")
    all_parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/mps)")
    all_parser.add_argument("--resume", action="store_true", help="Resume training")
    all_parser.add_argument("--pretrained", type=str, help="Pretrained model path")
    all_parser.add_argument("--data", type=str, help="Not used for 'all' command")

    args = parser.parse_args()

    if args.command == "setup":
        setup_data(args)
    elif args.command == "board":
        train_board(args)
    elif args.command == "pieces":
        train_pieces(args)
    elif args.command == "all":
        train_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
