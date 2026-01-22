"""
Dataset extraction and setup script.
Extracts board segmentation and pieces detection datasets from zip files.
"""

import zipfile
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BOARD_SEG_ZIP,
    PIECES_DET_ZIP,
    BOARD_SEG_DATA,
    PIECES_DATA,
    DATA_DIR,
)


def extract_dataset(zip_path: Path, output_dir: Path, name: str) -> bool:
    """
    Extract a dataset from zip file.

    Args:
        zip_path: Path to the zip file
        output_dir: Directory to extract to
        name: Name of the dataset for logging

    Returns:
        True if successful, False otherwise
    """
    if not zip_path.exists():
        print(f"Error: {name} zip file not found at {zip_path}")
        return False

    print(f"Extracting {name}...")
    print(f"  Source: {zip_path}")
    print(f"  Destination: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"  Successfully extracted {name}")
        return True
    except Exception as e:
        print(f"  Error extracting {name}: {e}")
        return False


def verify_dataset(dataset_dir: Path, name: str) -> dict:
    """
    Verify dataset structure after extraction.

    Args:
        dataset_dir: Path to the dataset directory
        name: Name of the dataset

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "name": name,
        "path": str(dataset_dir),
        "exists": dataset_dir.exists(),
        "splits": {},
    }

    if not dataset_dir.exists():
        return stats

    # Check for train/valid/test splits
    for split in ["train", "valid", "test"]:
        split_dir = dataset_dir / split
        if split_dir.exists():
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"

            image_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
            label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

            stats["splits"][split] = {
                "images": image_count,
                "labels": label_count,
            }

    # Check for data.yaml
    data_yaml = dataset_dir / "data.yaml"
    stats["has_data_yaml"] = data_yaml.exists()

    return stats


def print_dataset_stats(stats: dict):
    """Print formatted dataset statistics."""
    print(f"\nDataset: {stats['name']}")
    print(f"  Path: {stats['path']}")
    print(f"  Exists: {stats['exists']}")

    if stats['exists']:
        print(f"  Has data.yaml: {stats['has_data_yaml']}")
        for split, counts in stats['splits'].items():
            print(f"  {split.capitalize()}: {counts['images']} images, {counts['labels']} labels")


def setup_datasets():
    """Main function to setup all datasets."""
    print("=" * 60)
    print("Xiangqi Recognition - Dataset Setup")
    print("=" * 60)

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Extract board segmentation dataset
    if BOARD_SEG_ZIP.exists():
        success = extract_dataset(BOARD_SEG_ZIP, BOARD_SEG_DATA, "Board Segmentation Dataset")
        results.append(("Board Segmentation", success))
    else:
        print(f"\nWarning: Board segmentation zip not found at {BOARD_SEG_ZIP}")
        results.append(("Board Segmentation", False))

    # Extract pieces detection dataset
    if PIECES_DET_ZIP.exists():
        success = extract_dataset(PIECES_DET_ZIP, PIECES_DATA, "Pieces Detection Dataset")
        results.append(("Pieces Detection", success))
    else:
        print(f"\nWarning: Pieces detection zip not found at {PIECES_DET_ZIP}")
        results.append(("Pieces Detection", False))

    # Verify and print statistics
    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)

    board_stats = verify_dataset(BOARD_SEG_DATA, "Board Segmentation")
    print_dataset_stats(board_stats)

    pieces_stats = verify_dataset(PIECES_DATA, "Pieces Detection")
    print_dataset_stats(pieces_stats)

    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    return all(success for _, success in results)


if __name__ == "__main__":
    success = setup_datasets()
    sys.exit(0 if success else 1)
