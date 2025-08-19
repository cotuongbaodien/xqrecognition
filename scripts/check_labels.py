import argparse
import os
from glob import glob
from pathlib import Path

root_dir = Path(__file__).parent.parent
base_dir = root_dir / "dataset"

def check_labels(target):
    # check images and labels
    images_dir = base_dir / target / "images"
    labels_dir = base_dir / target / "labels"
    images = {os.path.splitext(os.path.basename(f))[0] for f in glob(f"{images_dir}/*")}
    labels = {os.path.splitext(os.path.basename(f))[0] for f in glob(f"{labels_dir}/*")}
    missing_labels = images - labels
    if missing_labels:
        print("Images missing labels:", missing_labels)
    missing_images = labels - images
    if missing_images:
        print("Labels missing images:", missing_images)

    # check labels
    for label_file in glob(f"{labels_dir}/*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        if not lines:
            print(f"Empty label file: {label_file}")
            continue

        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"{label_file} (line {i}) has wrong number of values: {parts}")
                continue

            try:
                int(parts[0])
                x, y, w, h = map(float, parts[1:])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    print(f"{label_file} (line {i}) out-of-range: {parts}")
            except ValueError:
                print(f"{label_file} (line {i}) contains invalid characters: {parts}")

    print("Checked!")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check labels of YOLO dataset')
    parser.add_argument('target', choices=['train', 'valid', 'test'], help='Target to check')
    args = parser.parse_args()
    check_labels(args.target)