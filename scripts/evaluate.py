"""
Evaluation script for Xiangqi Recognition System.
Evaluates the full pipeline on test images.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from config.settings import (
    BOARD_SEG_MODEL,
    PIECES_DET_MODEL,
    PIECES_DATA,
)
from src.pipeline import XiangqiRecognizer


def evaluate_pieces_model(
    model_path: str = None,
    data_dir: str = None,
    split: str = "test"
) -> Dict:
    """
    Evaluate the pieces detection model using YOLO's built-in evaluation.

    Args:
        model_path: Path to the model.
        data_dir: Path to the dataset directory.
        split: Dataset split to evaluate ('test', 'valid').

    Returns:
        Dictionary with evaluation metrics.
    """
    model_path = model_path or str(PIECES_DET_MODEL)

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return {}

    print("=" * 60)
    print("Pieces Detection Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")

    model = YOLO(model_path)

    # Find data.yaml
    data_dir = Path(data_dir) if data_dir else PIECES_DATA
    data_yaml = None

    for candidate in [data_dir / "data.yaml", data_dir / "dataset.yaml"]:
        if candidate.exists():
            data_yaml = candidate
            break

    # Also check subdirectories
    if data_yaml is None:
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                for candidate in [subdir / "data.yaml", subdir / "dataset.yaml"]:
                    if candidate.exists():
                        data_yaml = candidate
                        break

    if data_yaml is None:
        print(f"Error: Could not find data.yaml in {data_dir}")
        return {}

    print(f"Dataset: {data_yaml}")
    print(f"Split: {split}")

    # Run validation
    results = model.val(data=str(data_yaml), split=split)

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    print("\nResults:")
    print(f"  mAP@50: {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

    return metrics


def evaluate_board_model(
    model_path: str = None,
    data_dir: str = None,
    split: str = "test"
) -> Dict:
    """
    Evaluate the board segmentation model.

    Args:
        model_path: Path to the model.
        data_dir: Path to the dataset directory.
        split: Dataset split to evaluate.

    Returns:
        Dictionary with evaluation metrics.
    """
    from config.settings import BOARD_SEG_DATA

    model_path = model_path or str(BOARD_SEG_MODEL)

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return {}

    print("=" * 60)
    print("Board Segmentation Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")

    model = YOLO(model_path)

    # Find data.yaml
    data_dir = Path(data_dir) if data_dir else BOARD_SEG_DATA
    data_yaml = None

    for candidate in [data_dir / "data.yaml", data_dir / "dataset.yaml"]:
        if candidate.exists():
            data_yaml = candidate
            break

    if data_yaml is None:
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                for candidate in [subdir / "data.yaml", subdir / "dataset.yaml"]:
                    if candidate.exists():
                        data_yaml = candidate
                        break

    if data_yaml is None:
        print(f"Error: Could not find data.yaml in {data_dir}")
        return {}

    print(f"Dataset: {data_yaml}")
    print(f"Split: {split}")

    # Run validation
    results = model.val(data=str(data_yaml), split=split)

    metrics = {
        "mAP50": float(results.seg.map50) if hasattr(results, 'seg') else float(results.box.map50),
        "mAP50-95": float(results.seg.map) if hasattr(results, 'seg') else float(results.box.map),
    }

    print("\nResults:")
    print(f"  mAP@50: {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")

    return metrics


def evaluate_pipeline(
    test_dir: str,
    ground_truth_file: str = None,
    output_file: str = None
) -> Dict:
    """
    Evaluate the full recognition pipeline.

    Args:
        test_dir: Directory containing test images.
        ground_truth_file: JSON file with ground truth FEN for each image.
        output_file: Output file for results.

    Returns:
        Dictionary with evaluation metrics.
    """
    print("=" * 60)
    print("Full Pipeline Evaluation")
    print("=" * 60)

    test_dir = Path(test_dir)
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return {}

    # Load ground truth if available
    ground_truth = {}
    if ground_truth_file and Path(ground_truth_file).exists():
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)

    # Find test images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    test_images = [
        f for f in test_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not test_images:
        print(f"Error: No images found in {test_dir}")
        return {}

    print(f"Found {len(test_images)} test images")
    print(f"Ground truth: {'Available' if ground_truth else 'Not available'}")

    # Initialize recognizer
    try:
        recognizer = XiangqiRecognizer(use_board_detection=True)
    except Exception as e:
        print(f"Warning: {e}")
        print("Trying without board detection...")
        recognizer = XiangqiRecognizer(use_board_detection=False)

    # Run evaluation
    results = []
    correct_fen = 0
    total_pieces = 0
    correct_pieces = 0

    for img_path in test_images:
        try:
            result = recognizer.recognize(str(img_path))
            img_name = img_path.name

            result_entry = {
                "image": img_name,
                "predicted_fen": result.fen,
                "piece_count": len(result.pieces),
                "confidence": result.confidence,
                "errors": result.errors,
            }

            # Compare with ground truth if available
            if img_name in ground_truth:
                gt_fen = ground_truth[img_name]
                result_entry["ground_truth_fen"] = gt_fen
                result_entry["fen_match"] = result.fen == gt_fen

                if result.fen == gt_fen:
                    correct_fen += 1

                # Calculate piece-level accuracy
                from src.fen_generator import FENGenerator
                fen_gen = FENGenerator()
                comparison = fen_gen.compare_fen(gt_fen, result.fen)
                result_entry["piece_accuracy"] = comparison["accuracy"]
                total_pieces += comparison["total_pieces"]
                correct_pieces += comparison["matching"]

            results.append(result_entry)
            print(f"  {img_name}: {len(result.pieces)} pieces, conf={result.confidence:.2f}")

        except Exception as e:
            print(f"  {img_path.name}: Error - {e}")
            results.append({
                "image": img_path.name,
                "error": str(e),
            })

    # Calculate metrics
    metrics = {
        "total_images": len(test_images),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
    }

    if ground_truth:
        metrics["fen_accuracy"] = correct_fen / len(ground_truth) if ground_truth else 0
        metrics["piece_accuracy"] = correct_pieces / total_pieces if total_pieces > 0 else 0

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total images: {metrics['total_images']}")
    print(f"Successful: {metrics['successful']}")
    print(f"Failed: {metrics['failed']}")

    if ground_truth:
        print(f"FEN accuracy: {metrics['fen_accuracy']:.2%}")
        print(f"Piece accuracy: {metrics['piece_accuracy']:.2%}")

    # Save results
    if output_file:
        output = {
            "metrics": metrics,
            "results": results,
        }
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Xiangqi Recognition System")

    subparsers = parser.add_subparsers(dest="command", help="Evaluation command")

    # Pieces model evaluation
    pieces_parser = subparsers.add_parser("pieces", help="Evaluate pieces model")
    pieces_parser.add_argument("--model", type=str, help="Path to model")
    pieces_parser.add_argument("--data", type=str, help="Path to dataset")
    pieces_parser.add_argument("--split", type=str, default="test", help="Dataset split")

    # Board model evaluation
    board_parser = subparsers.add_parser("board", help="Evaluate board model")
    board_parser.add_argument("--model", type=str, help="Path to model")
    board_parser.add_argument("--data", type=str, help="Path to dataset")
    board_parser.add_argument("--split", type=str, default="test", help="Dataset split")

    # Full pipeline evaluation
    pipeline_parser = subparsers.add_parser("pipeline", help="Evaluate full pipeline")
    pipeline_parser.add_argument("--test-dir", type=str, required=True,
                                  help="Directory with test images")
    pipeline_parser.add_argument("--ground-truth", type=str,
                                  help="JSON file with ground truth FEN")
    pipeline_parser.add_argument("--output", type=str,
                                  help="Output file for results")

    # All evaluation
    all_parser = subparsers.add_parser("all", help="Run all evaluations")
    all_parser.add_argument("--test-dir", type=str, help="Directory with test images")

    args = parser.parse_args()

    if args.command == "pieces":
        evaluate_pieces_model(args.model, args.data, args.split)
    elif args.command == "board":
        evaluate_board_model(args.model, args.data, args.split)
    elif args.command == "pipeline":
        evaluate_pipeline(args.test_dir, args.ground_truth, args.output)
    elif args.command == "all":
        print("\n--- Pieces Model Evaluation ---")
        evaluate_pieces_model()
        print("\n--- Board Model Evaluation ---")
        evaluate_board_model()
        if args.test_dir:
            print("\n--- Pipeline Evaluation ---")
            evaluate_pipeline(args.test_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
