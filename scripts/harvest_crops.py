"""Harvest REAL piece crops from a labeled YOLO dataset (e.g. items_v16, ~4k).

For every piece bbox (the 14 PIECE_CLASS_IDS, landmarks skipped) cut the region,
bake a feathered ELLIPTICAL alpha (Xiangqi pieces are round discs, so an
inscribed ellipse removes the board-background corners), and save an RGBA PNG
under data/piece_crops_real/<class_name>/. These real crops carry true wood/
plastic texture + lighting and are mixed with the digital skins by synth_gen
to close the sim->real gap.

Usage:
    python scripts/harvest_crops.py --src data/items_v16 \
        --out data/piece_crops_real --per-class 600 --min-px 24
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from boarddetection.settings import ITEM_CLASSES, PIECE_CLASS_IDS  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def elliptical_rgba(crop_bgr, feather=0.12):
    """Return RGBA where alpha is a feathered inscribed ellipse of the crop."""
    h, w = crop_bgr.shape[:2]
    alpha = np.zeros((h, w), np.uint8)
    cv2.ellipse(alpha, (w // 2, h // 2),
                (int(w * 0.49), int(h * 0.49)), 0, 0, 360, 255, -1)
    k = max(1, int(round(min(h, w) * feather)) | 1)  # odd kernel
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    rgba = np.dstack([crop_bgr, alpha])
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/items_v16")
    ap.add_argument("--out", default="data/piece_crops_real")
    ap.add_argument("--per-class", type=int, default=600,
                    help="max crops saved per class")
    ap.add_argument("--min-px", type=int, default=24,
                    help="skip bbox smaller than this (px, shorter side)")
    ap.add_argument("--pad", type=float, default=0.06,
                    help="bbox padding fraction before crop")
    args = ap.parse_args()

    src = PROJECT_ROOT / args.src
    out = PROJECT_ROOT / args.out
    if out.exists():
        import shutil
        shutil.rmtree(out)
    id2name = {i: ITEM_CLASSES[i][0] for i in PIECE_CLASS_IDS}
    for name in id2name.values():
        (out / name).mkdir(parents=True, exist_ok=True)

    counts = {i: 0 for i in PIECE_CLASS_IDS}
    n_img = 0
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / (img_path.stem + ".txt")
            if not lbl.exists():
                continue
            if all(counts[i] >= args.per_class for i in PIECE_CLASS_IDS):
                break
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            n_img += 1
            H, W = img.shape[:2]
            for li, line in enumerate(lbl.read_text().splitlines()):
                p = line.split()
                if len(p) != 5:
                    continue
                cid = int(p[0])
                if cid not in PIECE_CLASS_IDS or counts[cid] >= args.per_class:
                    continue
                cx, cy, bw, bh = (float(p[1]) * W, float(p[2]) * H,
                                  float(p[3]) * W, float(p[4]) * H)
                pad = args.pad
                x1 = int(cx - bw * (0.5 + pad)); y1 = int(cy - bh * (0.5 + pad))
                x2 = int(cx + bw * (0.5 + pad)); y2 = int(cy + bh * (0.5 + pad))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 < args.min_px or y2 - y1 < args.min_px:
                    continue
                crop = img[y1:y2, x1:x2]
                rgba = elliptical_rgba(crop)
                name = id2name[cid]
                cv2.imwrite(str(out / name / f"{img_path.stem}_{li}.png"), rgba)
                counts[cid] += 1

    print(f"Harvested from {n_img} images:")
    total = 0
    for i in sorted(PIECE_CLASS_IDS):
        print(f"  {id2name[i]:16s} {counts[i]}")
        total += counts[i]
    print(f"Total real crops: {total}\nOutput: {out}")


if __name__ == "__main__":
    main()
