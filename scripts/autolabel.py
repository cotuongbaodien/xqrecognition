"""Auto-label images with the current items model (v9, 18-class) for
model-assisted annotation.

Runs items.pt on every image in --src (recursively), writes YOLO detection
labels (18 classes: 14 pieces + board-conner + 3 palace types), and copies
images alongside. Upload the result to a fresh Roboflow project and correct
the pre-labels — far faster than annotating from scratch.

Usage:
    python scripts/autolabel.py --src data/items_v9 --out data/items_v9_autolabel --conf 0.3
"""

import argparse
import shutil
from pathlib import Path

import cv2

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from boarddetection.item_detector import ItemDetector
from boarddetection.settings import ITEM_CLASSES, ITEMS_MODEL

CLASS_NAMES = [ITEM_CLASSES[i][0] for i in range(len(ITEM_CLASSES))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/items_v9", help="Folder of images (recursive)")
    ap.add_argument("--out", default="data/items_v9_autolabel")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    src = PROJECT_ROOT / args.src
    out = PROJECT_ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)

    det = ItemDetector(args.model or str(ITEMS_MODEL))

    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    print(f"Auto-labeling {len(imgs)} images @ conf={args.conf} ...")

    n_done = 0
    n_boxes = 0
    seen = set()
    for p in imgs:
        # de-dup by filename stem (datasets may repeat across splits)
        if p.stem in seen:
            continue
        seen.add(p.stem)
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        result = det.detect(img, confidence=args.conf)

        lines = []
        # pieces
        for pc in result.pieces:
            x1, y1, x2, y2 = pc.bbox
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{pc.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        # landmarks
        for name in ("board-conner", "palace-bottom", "palace-center", "palace-conner"):
            cls_id = CLASS_NAMES.index(name)
            for lm in result.get_landmarks(name):
                x1, y1, x2, y2 = lm.bbox
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        n_boxes += len(lines)
        shutil.copy(p, out / "images" / p.name)
        (out / "labels" / f"{p.stem}.txt").write_text("\n".join(lines))
        n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done}...")

    # data.yaml
    names = ", ".join(f"'{c}'" for c in CLASS_NAMES)
    (out / "data.yaml").write_text(
        f"train: images\nval: images\n\nnc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n", encoding="utf-8"
    )
    print(f"\nDone: {n_done} images, {n_boxes} boxes ({n_boxes/max(1,n_done):.1f}/img)")
    print(f"Output: {out}  → upload to Roboflow, correct pre-labels.")


if __name__ == "__main__":
    main()
