"""
Create corner detection dataset from board segmentation data.
Extracts the 4 corner points from intersection annotations.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
BOARD_SEG_DATA = PROJECT_ROOT / "data" / "board_seg"
CORNER_DATA = PROJECT_ROOT / "data" / "board_corners"


def parse_segmentation_label(label_path: Path) -> list:
    """Parse YOLO segmentation label and extract polygon centroids."""
    centroids = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # Class ID is first, then x,y pairs
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # Extract x,y pairs
            xs = coords[0::2]
            ys = coords[1::2]

            if len(xs) > 0 and len(ys) > 0:
                # Calculate centroid
                cx = np.mean(xs)
                cy = np.mean(ys)
                centroids.append((cx, cy))

    return centroids


def find_corners(centroids: list) -> dict:
    """Find the 4 corner points from a list of centroids."""
    if len(centroids) < 4:
        return None

    points = np.array(centroids)

    # Find corners using sum and difference of coordinates
    # Top-left: min(x + y)
    # Top-right: max(x - y)
    # Bottom-right: max(x + y)
    # Bottom-left: min(x - y)

    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]

    top_left_idx = np.argmin(sums)
    bottom_right_idx = np.argmax(sums)
    top_right_idx = np.argmax(diffs)
    bottom_left_idx = np.argmin(diffs)

    return {
        'top_left': points[top_left_idx].tolist(),
        'top_right': points[top_right_idx].tolist(),
        'bottom_right': points[bottom_right_idx].tolist(),
        'bottom_left': points[bottom_left_idx].tolist(),
    }


def create_yolo_keypoint_label(corners: dict) -> str:
    """
    Create YOLO keypoint format label.
    Format: class_id x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    where v is visibility (2 = visible)
    """
    # Get all corner coordinates
    tl = corners['top_left']
    tr = corners['top_right']
    br = corners['bottom_right']
    bl = corners['bottom_left']

    # Calculate bounding box
    all_x = [tl[0], tr[0], br[0], bl[0]]
    all_y = [tl[1], tr[1], br[1], bl[1]]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # Create label: class_id cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v kp3_x kp3_y kp3_v kp4_x kp4_y kp4_v
    # Keypoint order: top_left, top_right, bottom_right, bottom_left
    label = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} "
    label += f"{tl[0]:.6f} {tl[1]:.6f} 2 "
    label += f"{tr[0]:.6f} {tr[1]:.6f} 2 "
    label += f"{br[0]:.6f} {br[1]:.6f} 2 "
    label += f"{bl[0]:.6f} {bl[1]:.6f} 2"

    return label


def create_data_yaml(output_dir: Path):
    """Create data.yaml for YOLO keypoint training."""
    data = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'test/images',
        'names': {0: 'board'},
        'kpt_shape': [4, 3],  # 4 keypoints, 3 values each (x, y, visibility)
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Created {yaml_path}")


def process_split(split: str, input_dir: Path, output_dir: Path):
    """Process a single split (train/test/valid)."""
    labels_dir = input_dir / split / "labels"
    images_dir = input_dir / split / "images"

    if not labels_dir.exists():
        print(f"Skipping {split}: {labels_dir} not found")
        return 0

    # Create output directories
    out_labels = output_dir / split / "labels"
    out_images = output_dir / split / "images"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    processed = 0

    for label_file in labels_dir.glob("*.txt"):
        # Parse segmentation label
        centroids = parse_segmentation_label(label_file)

        if len(centroids) < 4:
            print(f"Skipping {label_file.name}: only {len(centroids)} points")
            continue

        # Find corners
        corners = find_corners(centroids)
        if corners is None:
            continue

        # Create keypoint label
        keypoint_label = create_yolo_keypoint_label(corners)

        # Save label
        out_label_path = out_labels / label_file.name
        with open(out_label_path, 'w') as f:
            f.write(keypoint_label + '\n')

        # Copy corresponding image
        base_name = label_file.stem
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_path = images_dir / (base_name + ext)
            if img_path.exists():
                shutil.copy(img_path, out_images / img_path.name)
                break

        processed += 1

    print(f"Processed {processed} images for {split}")
    return processed


def main():
    print("=" * 60)
    print("Creating Corner Detection Dataset")
    print("=" * 60)

    # Clean output directory
    if CORNER_DATA.exists():
        shutil.rmtree(CORNER_DATA)
    CORNER_DATA.mkdir(parents=True)

    total = 0
    for split in ['train', 'valid', 'test']:
        total += process_split(split, BOARD_SEG_DATA, CORNER_DATA)

    # Create data.yaml
    create_data_yaml(CORNER_DATA)

    print("=" * 60)
    print(f"Total images processed: {total}")
    print(f"Dataset saved to: {CORNER_DATA}")
    print("=" * 60)


if __name__ == "__main__":
    main()
