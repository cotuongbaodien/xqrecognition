"""Auto-generate board-segmentation polygon labels from existing detection
labels.

Existing detection labels already mark the board perimeter via
board-conner (id 8), board-border (id 7), and palace-bottom (id 9) —
all of which lie on the 9x10 grid boundary. Their convex hull
approximates the board outline. This script converts each detection
label file into a YOLO-segmentation polygon label (single class
'xiangqi-board') so a board-seg model can be trained on the SAME images
without hand-labeling polygons from scratch.

Output is a starting point — review/refine on Roboflow, especially for
images with sparse landmark detections (heavy occlusion).

Usage:
    python scripts/gen_board_polygon.py --src data/items_v7 --out data/board_seg
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

# Perimeter landmark class IDs in the v6+ detection scheme
PERIMETER_IDS = {7, 8, 9}  # board-border, board-conner, palace-bottom
MIN_POINTS = 4             # need at least this many to form a board hull


def hull_polygon(points):
    """Convex hull of normalized (x,y) points → ordered polygon vertices.
    Points and output are in normalized [0,1] coords."""
    pts = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(pts)
    return [(float(p[0][0]), float(p[0][1])) for p in hull]


def parse_perimeter(label_path):
    """Read detection label, return normalized centers of perimeter landmarks."""
    pts = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(parts[0])
        if cls_id in PERIMETER_IDS:
            cx, cy = float(parts[1]), float(parts[2])
            pts.append((cx, cy))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/items_v7",
                    help="Source detection dataset (has train/valid/test splits)")
    ap.add_argument("--out", default="data/board_seg",
                    help="Output segmentation dataset dir")
    ap.add_argument("--min-points", type=int, default=MIN_POINTS,
                    help="Skip images with fewer perimeter landmarks")
    ap.add_argument("--min-extent", type=float, default=0.35,
                    help="Min normalized hull span (w and h) to accept; "
                         "rejects clustered-point hulls")
    args = ap.parse_args()

    src = PROJECT_ROOT / args.src
    out = PROJECT_ROOT / args.out
    if out.exists():
        shutil.rmtree(out)

    n_ok = 0
    n_skip = 0
    n_total = 0
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.exists():
            continue
        out_img = out / split / "images"
        out_lbl = out / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            n_total += 1
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                n_skip += 1
                continue
            pts = parse_perimeter(lbl_path)
            if len(pts) < args.min_points:
                n_skip += 1
                continue
            poly = hull_polygon(pts)
            if len(poly) < 3:
                n_skip += 1
                continue
            # Spatial-extent gate: reject hulls that don't span enough of the
            # frame — happens when perimeter points cluster (sparse detection
            # near pieces only), giving a tiny wrong polygon instead of the
            # full board outline.
            xs = [x for x, _ in poly]
            ys = [y for _, y in poly]
            if (max(xs) - min(xs)) < args.min_extent or \
               (max(ys) - min(ys)) < args.min_extent:
                n_skip += 1
                continue
            # YOLO-seg line: class_id followed by flattened normalized polygon
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            (out_lbl / (img_path.stem + ".txt")).write_text(f"0 {coords}\n")
            shutil.copy(img_path, out_img / img_path.name)
            n_ok += 1

    # data.yaml for segmentation (single class)
    yaml_text = (
        f"path: {out.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: xiangqi-board\n"
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Total images:     {n_total}")
    print(f"Polygons written: {n_ok}")
    print(f"Skipped (sparse): {n_skip}")
    print(f"\nOutput: {out}")
    print("Next: review/refine polygons on Roboflow, then train:")
    print(f"  yolo segment train data={args.out}/data.yaml model=yolo11n-seg.pt epochs=100")


if __name__ == "__main__":
    main()
