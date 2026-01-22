"""
Detection CLI for Xiangqi Recognition System.
Provides command-line interface for detecting pieces and generating FEN from images.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import BOARD_SEG_MODEL, PIECES_DET_MODEL


def detect_image(
    image_path: str,
    output_dir: str = None,
    visualize: bool = True,
    board_model: str = None,
    pieces_model: str = None,
    confidence: float = 0.5,
    use_board: bool = True,
) -> dict:
    """
    Detect pieces in a single image.

    Args:
        image_path: Path to the image.
        output_dir: Directory to save results.
        visualize: Whether to save visualization.
        board_model: Path to board model.
        pieces_model: Path to pieces model.
        confidence: Confidence threshold.
        use_board: Whether to use board detection.

    Returns:
        Detection result dictionary.
    """
    from src.pipeline import XiangqiRecognizer

    # Initialize recognizer
    board_path = board_model or str(BOARD_SEG_MODEL)
    pieces_path = pieces_model or str(PIECES_DET_MODEL)

    use_board = use_board and Path(board_path).exists()

    recognizer = XiangqiRecognizer(
        board_model_path=board_path if use_board else None,
        pieces_model_path=pieces_path,
        use_board_detection=use_board,
    )

    # Run detection
    result = recognizer.recognize(
        image_path,
        piece_confidence=confidence,
        visualize=visualize,
    )

    # Prepare output
    output = result.to_dict()
    output["image"] = str(image_path)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON result
        json_path = output_dir / f"{Path(image_path).stem}_result.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved result to: {json_path}")

        # Save visualization
        if visualize and result.visualization is not None:
            vis_path = output_dir / f"{Path(image_path).stem}_visualization.png"
            cv2.imwrite(str(vis_path), result.visualization)
            print(f"Saved visualization to: {vis_path}")

    return output


def detect_directory(
    input_dir: str,
    output_dir: str = None,
    visualize: bool = True,
    board_model: str = None,
    pieces_model: str = None,
    confidence: float = 0.5,
    use_board: bool = True,
) -> list:
    """
    Detect pieces in all images in a directory.

    Args:
        input_dir: Directory containing images.
        output_dir: Directory to save results.
        visualize: Whether to save visualizations.
        board_model: Path to board model.
        pieces_model: Path to pieces model.
        confidence: Confidence threshold.
        use_board: Whether to use board detection.

    Returns:
        List of detection results.
    """
    from src.pipeline import XiangqiRecognizer

    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return []

    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in: {input_dir}")
        return []

    print(f"Found {len(images)} images")

    # Initialize recognizer
    board_path = board_model or str(BOARD_SEG_MODEL)
    pieces_path = pieces_model or str(PIECES_DET_MODEL)

    use_board = use_board and Path(board_path).exists()

    recognizer = XiangqiRecognizer(
        board_model_path=board_path if use_board else None,
        pieces_model_path=pieces_path,
        use_board_detection=use_board,
    )

    # Process images
    results = []
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing: {image_path.name}")

        try:
            result = recognizer.recognize(
                str(image_path),
                piece_confidence=confidence,
                visualize=visualize,
            )

            output = result.to_dict()
            output["image"] = str(image_path)
            results.append(output)

            print(f"  FEN: {result.fen}")
            print(f"  Pieces: {len(result.pieces)}, Confidence: {result.confidence:.2f}")

            # Save visualization
            if output_dir and visualize and result.visualization is not None:
                vis_path = output_dir / f"{image_path.stem}_visualization.png"
                cv2.imwrite(str(vis_path), result.visualization)

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "image": str(image_path),
                "error": str(e),
            })

    # Save summary JSON
    if output_dir:
        summary_path = output_dir / "results.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved summary to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Xiangqi Recognition System - Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect from single image
  python detect.py --image board.jpg

  # Detect from directory
  python detect.py --dir images/

  # Save results to output directory
  python detect.py --image board.jpg --output output/

  # Disable board detection (faster, less accurate)
  python detect.py --image board.jpg --no-board

  # Custom confidence threshold
  python detect.py --image board.jpg --confidence 0.3
        """
    )

    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--dir", type=str, help="Path to directory of images")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--board-model", type=str, help="Path to board segmentation model")
    parser.add_argument("--pieces-model", type=str, help="Path to pieces detection model")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--no-board", action="store_true", help="Disable board detection")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")

    args = parser.parse_args()

    if args.image:
        # Single image detection
        result = detect_image(
            args.image,
            output_dir=args.output,
            visualize=not args.no_visualize,
            board_model=args.board_model,
            pieces_model=args.pieces_model,
            confidence=args.confidence,
            use_board=not args.no_board,
        )

        print("\n" + "=" * 60)
        print("Detection Result")
        print("=" * 60)
        print(f"Image: {args.image}")
        print(f"FEN: {result.get('fen', 'N/A')}")
        print(f"Pieces detected: {result.get('piece_count', 0)}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")

        if result.get('errors'):
            print(f"Warnings: {', '.join(result['errors'])}")

    elif args.dir:
        # Directory detection
        results = detect_directory(
            args.dir,
            output_dir=args.output,
            visualize=not args.no_visualize,
            board_model=args.board_model,
            pieces_model=args.pieces_model,
            confidence=args.confidence,
            use_board=not args.no_board,
        )

        # Summary
        print("\n" + "=" * 60)
        print("Detection Summary")
        print("=" * 60)
        successful = len([r for r in results if 'error' not in r])
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
