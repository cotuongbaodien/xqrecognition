"""
Merge multiple pieces datasets into one for training.
Handles class ID remapping between different datasets.
"""

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Dataset paths
DATASET_OLD = PROJECT_ROOT / "data" / "pieces"  # Chinese-chess.v9i.yolov8
DATASET_NEW = PROJECT_ROOT / "data" / "xiangqi_pieces"  # xiangqi.v6i.yolov8
DATASET_V4 = PROJECT_ROOT / "data" / "xiangqi_v4"  # XiangQi.v4i.yolov8
DATASET_MERGED = PROJECT_ROOT / "data" / "pieces_merged"

# Class mapping from new dataset to old dataset class IDs
# Old dataset classes (target):
# 0: Advisor_black, 1: Advisor_red, 2: Cannon_black, 3: Cannon_red,
# 4: Elephant_black, 5: Elephant_red, 6: General_black, 7: General_red,
# 8: Knight_black, 9: Knight_red, 10: Pawn_black, 11: Pawn_red,
# 12: Rook_black, 13: Rook_red

# New dataset classes (source):
# 0: black_assistant, 1: black_bishop, 2: black_chariot, 3: black_king,
# 4: black_knight, 5: black_pawn, 6: black_rook,
# 7: red_assistant, 8: red_bishop, 9: red_chariot, 10: red_king,
# 11: red_knight, 12: red_pawn, 13: red_rook

NEW_TO_OLD_MAPPING = {
    0: 0,   # black_assistant -> Advisor_black
    1: 4,   # black_bishop -> Elephant_black
    2: 2,   # black_chariot -> Cannon_black
    3: 6,   # black_king -> General_black
    4: 8,   # black_knight -> Knight_black
    5: 10,  # black_pawn -> Pawn_black
    6: 12,  # black_rook -> Rook_black
    7: 1,   # red_assistant -> Advisor_red
    8: 5,   # red_bishop -> Elephant_red
    9: 3,   # red_chariot -> Cannon_red
    10: 7,  # red_king -> General_red
    11: 9,  # red_knight -> Knight_red
    12: 11, # red_pawn -> Pawn_red
    13: 13, # red_rook -> Rook_red
}

# XiangQi.v4i.yolov8 class mapping
# 0: black_advisor, 1: black_cannon, 2: black_chariot, 3: black_elephant,
# 4: black_general, 5: black_horse, 6: black_soldier,
# 7: red_advisor, 8: red_cannon, 9: red_chariot, 10: red_elephant,
# 11: red_general, 12: red_horse, 13: red_soldier
V4_TO_OLD_MAPPING = {
    0: 0,   # black_advisor -> Advisor_black
    1: 2,   # black_cannon -> Cannon_black
    2: 12,  # black_chariot -> Rook_black
    3: 4,   # black_elephant -> Elephant_black
    4: 6,   # black_general -> General_black
    5: 8,   # black_horse -> Knight_black
    6: 10,  # black_soldier -> Pawn_black
    7: 1,   # red_advisor -> Advisor_red
    8: 3,   # red_cannon -> Cannon_red
    9: 13,  # red_chariot -> Rook_red
    10: 5,  # red_elephant -> Elephant_red
    11: 7,  # red_general -> General_red
    12: 9,  # red_horse -> Knight_red
    13: 11, # red_soldier -> Pawn_red
}


def remap_label_file(src_path: Path, dst_path: Path, class_mapping: dict):
    """Remap class IDs in a YOLO label file."""
    with open(src_path, 'r') as f:
        lines = f.readlines()

    remapped_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_class_id = int(parts[0])
            new_class_id = class_mapping.get(old_class_id, old_class_id)
            parts[0] = str(new_class_id)
            remapped_lines.append(' '.join(parts) + '\n')

    with open(dst_path, 'w') as f:
        f.writelines(remapped_lines)


def copy_dataset(src_dir: Path, dst_dir: Path, class_mapping: dict = None, prefix: str = ""):
    """Copy a dataset split, optionally remapping class IDs."""
    for split in ['train', 'valid', 'test']:
        src_images = src_dir / split / 'images'
        src_labels = src_dir / split / 'labels'
        dst_images = dst_dir / split / 'images'
        dst_labels = dst_dir / split / 'labels'

        if not src_images.exists():
            continue

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img_file in src_images.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dst_name = f"{prefix}{img_file.name}" if prefix else img_file.name
                shutil.copy(img_file, dst_images / dst_name)

        # Copy/remap labels
        if src_labels.exists():
            for label_file in src_labels.glob('*.txt'):
                dst_name = f"{prefix}{label_file.name}" if prefix else label_file.name
                dst_path = dst_labels / dst_name

                if class_mapping:
                    remap_label_file(label_file, dst_path, class_mapping)
                else:
                    shutil.copy(label_file, dst_path)


def create_data_yaml(output_dir: Path):
    """Create data.yaml for the merged dataset."""
    yaml_content = f"""path: {output_dir.absolute()}
train: train/images
val: valid/images
test: test/images

nc: 14
names:
  0: Advisor_black
  1: Advisor_red
  2: Cannon_black
  3: Cannon_red
  4: Elephant_black
  5: Elephant_red
  6: General_black
  7: General_red
  8: Knight_black
  9: Knight_red
  10: Pawn_black
  11: Pawn_red
  12: Rook_black
  13: Rook_red
"""

    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"Created {output_dir / 'data.yaml'}")


def count_images(dataset_dir: Path) -> dict:
    """Count images in each split."""
    counts = {}
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_dir / split / 'images'
        if img_dir.exists():
            counts[split] = len(list(img_dir.glob('*')))
        else:
            counts[split] = 0
    return counts


def main():
    print("=" * 60)
    print("Merging Pieces Datasets")
    print("=" * 60)

    # Clean output directory
    if DATASET_MERGED.exists():
        shutil.rmtree(DATASET_MERGED)
    DATASET_MERGED.mkdir(parents=True)

    # Copy old dataset (no remapping needed)
    print(f"\nCopying old dataset from {DATASET_OLD}...")
    if DATASET_OLD.exists():
        copy_dataset(DATASET_OLD, DATASET_MERGED, class_mapping=None, prefix="old_")
        counts = count_images(DATASET_MERGED)
        print(f"  After old dataset: {counts}")
    else:
        print(f"  Warning: Old dataset not found at {DATASET_OLD}")

    # Copy new dataset with remapping
    print(f"\nCopying new dataset from {DATASET_NEW} with class remapping...")
    if DATASET_NEW.exists():
        copy_dataset(DATASET_NEW, DATASET_MERGED, class_mapping=NEW_TO_OLD_MAPPING, prefix="new_")
        counts = count_images(DATASET_MERGED)
        print(f"  After new dataset: {counts}")
    else:
        print(f"  Warning: New dataset not found at {DATASET_NEW}")

    # Copy v4 dataset with remapping
    print(f"\nCopying v4 dataset from {DATASET_V4} with class remapping...")
    if DATASET_V4.exists():
        copy_dataset(DATASET_V4, DATASET_MERGED, class_mapping=V4_TO_OLD_MAPPING, prefix="v4_")
        counts = count_images(DATASET_MERGED)
        print(f"  After v4 dataset: {counts}")
    else:
        print(f"  Warning: V4 dataset not found at {DATASET_V4}")

    # Create data.yaml
    create_data_yaml(DATASET_MERGED)

    # Final counts
    print("\n" + "=" * 60)
    print("Final dataset counts:")
    counts = count_images(DATASET_MERGED)
    for split, count in counts.items():
        print(f"  {split}: {count} images")
    print(f"\nMerged dataset saved to: {DATASET_MERGED}")
    print("=" * 60)


if __name__ == "__main__":
    main()
