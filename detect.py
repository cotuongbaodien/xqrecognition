"""
Detection CLI for Xiangqi Recognition System.
Provides command-line interface for detecting pieces and generating FEN from images.
"""

import argparse
import json
from pathlib import Path

import cv2

from boarddetection import ITEMS_MODEL


def detect_image(
    image_path: str,
    output_dir: str = None,
    visualize: bool = True,
    items_model: str = None,
    confidence: float = 0.3,
) -> dict:
    """Detect pieces in a single image."""
    from boarddetection import XiangqiRecognizer

    recognizer = XiangqiRecognizer(
        items_model_path=items_model or str(ITEMS_MODEL),
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
    items_model: str = None,
    confidence: float = 0.3,
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
    from boarddetection import XiangqiRecognizer

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

    recognizer = XiangqiRecognizer(
        items_model_path=items_model or str(ITEMS_MODEL),
    )
    from boarddetection import render_fen_ascii

    # Process images
    results = []
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {image_path.name}")

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
            print(render_fen_ascii(result.fen))

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
  python detect.py --image board.jpg
  python detect.py --dir images/ --output output/
  python detect.py --image board.jpg --confidence 0.3
        """
    )

    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--dir", type=str, help="Path to directory of images")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--items-model", type=str, help="Path to items detection model")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")

    args = parser.parse_args()

    if args.image:
        # Single image detection
        result = detect_image(
            args.image,
            output_dir=args.output,
            visualize=not args.no_visualize,
            items_model=args.items_model,
            confidence=args.confidence,
        )

        from boarddetection import render_fen_ascii
        print("\n" + "=" * 60)
        print("Detection Result")
        print("=" * 60)
        print(f"Image: {args.image}")
        print(f"FEN: {result.get('fen', 'N/A')}")
        print(f"Pieces detected: {result.get('piece_count', 0)}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        print()
        print(render_fen_ascii(result.get('fen', '9/9/9/9/9/9/9/9/9/9')))

        if result.get('errors'):
            print(f"\nWarnings: {', '.join(result['errors'])}")

    elif args.dir:
        # Directory detection
        results = detect_directory(
            args.dir,
            output_dir=args.output,
            visualize=not args.no_visualize,
            items_model=args.items_model,
            confidence=args.confidence,
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
