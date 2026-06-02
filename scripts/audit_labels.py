"""Model-assisted label audit.

Run the deployed items model over the training images, compare each
predicted box against the existing YOLO label by IoU, and report cells
where the model's CLASS disagrees with the label class. Those are the
mislabel candidates to fix on Roboflow.

    python scripts/audit_labels.py --data data/items_v11 --device cpu

Output: label_audit.csv (sorted by label-class), Roboflow-searchable
source name + cell + label-class vs model-class.
"""

import argparse
import csv
import re
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

LANDMARKS = {"board-conner", "palace-bottom", "palace-center", "palace-conner"}


def category(cls_name):
    return "landmark" if cls_name in LANDMARKS else "piece"


def iou(a, b):
    # boxes xyxy
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_label(lbl_path, w, h):
    """Return list of (cls, xyxy) from a YOLO txt (normalized cx cy w h)."""
    out = []
    if not lbl_path.exists():
        return out
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(x) for x in p[1:5])
        out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def src_name(fn):
    """Roboflow source name: strip the .rf.<hash> augmentation suffix."""
    return re.split(r"\.rf\.", fn)[0]


def cell_label(box, w, h):
    """Rough a-i / 0-9 grid cell from box center, just to locate the piece."""
    cx = (box[0] + box[2]) / 2 / w
    cy = (box[1] + box[3]) / 2 / h
    col = "abcdefghi"[min(8, max(0, int(cx * 9)))]
    row = min(9, max(0, int(cy * 10)))
    return f"{col}{row}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/items_v11", help="dataset dir with train/")
    ap.add_argument("--model", default="boarddetection/models/items.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--out", default="label_audit.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = ap.parse_args()

    model = YOLO(str(PROJECT_ROOT / args.model))
    names = model.names

    img_dir = PROJECT_ROOT / args.data / "train" / "images"
    lbl_dir = PROJECT_ROOT / args.data / "train" / "labels"
    imgs = sorted(img_dir.glob("*"))
    if args.limit:
        imgs = imgs[: args.limit]

    seen_src = set()
    rows = []
    n_pieces = 0
    n_wrong = 0
    n_missed = 0

    for i, img in enumerate(imgs):
        src = src_name(img.name)
        if src in seen_src:          # one representative per source is enough
            continue
        seen_src.add(src)

        r = model.predict(str(img), conf=args.conf, device=args.device, verbose=False)[0]
        h, w = r.orig_shape
        preds = [([*b.xyxy[0].tolist()], int(b.cls), float(b.conf)) for b in r.boxes]
        labels = load_label(lbl_dir / (img.stem + ".txt"), w, h)

        for lc, lbox in labels:
            n_pieces += 1
            best, best_iou = None, 0.0
            for pbox, pc, pconf in preds:
                v = iou(lbox, pbox)
                if v > best_iou:
                    best, best_iou = (pc, pconf), v
            if best and best_iou >= args.iou:
                if best[0] != lc:        # model sees a piece but calls it wrong
                    n_wrong += 1
                    rows.append({
                        "type": "WRONG_CLASS",
                        "category": category(names[lc]),
                        "label_class": names[lc],
                        "model_class": names[best[0]],
                        "model_conf": round(best[1], 2),
                        "iou": round(best_iou, 2),
                        "cell": cell_label(lbox, w, h),
                        "source_image": src,
                        "file": img.name,
                    })
            else:                        # label has a piece, model detected nothing
                n_missed += 1
                rows.append({
                    "type": "MODEL_MISSED",
                    "category": category(names[lc]),
                    "label_class": names[lc],
                    "model_class": "(none)",
                    "model_conf": "",
                    "iou": round(best_iou, 2),
                    "cell": cell_label(lbox, w, h),
                    "source_image": src,
                    "file": img.name,
                })
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(imgs)} imgs | {len(seen_src)} sources | "
                  f"{n_wrong} wrong-class, {n_missed} missed")

    # Split into 3 files by the user's fix priority:
    #   1. fix label   -> WRONG_CLASS on pieces
    #   2. add piece   -> MODEL_MISSED on pieces
    #   3. landmark    -> anything on landmark classes
    buckets = {
        "audit_1_fix_label.csv":   [r for r in rows if r["category"] == "piece" and r["type"] == "WRONG_CLASS"],
        "audit_2_add_piece.csv":   [r for r in rows if r["category"] == "piece" and r["type"] == "MODEL_MISSED"],
        "audit_3_landmark.csv":    [r for r in rows if r["category"] == "landmark"],
    }
    fields = ["type", "category", "label_class", "model_class", "model_conf",
              "iou", "cell", "source_image", "file"]
    from collections import Counter
    print(f"\n=== Audit done ===")
    print(f"Sources checked : {len(seen_src)} | pieces compared : {n_pieces}")
    for fname, bucket in buckets.items():
        bucket.sort(key=lambda x: (x["label_class"], -(x["model_conf"] or 0)))
        p = PROJECT_ROOT / fname
        with p.open("w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=fields)
            wri.writeheader()
            wri.writerows(bucket)
        c = Counter(r["label_class"] for r in bucket)
        print(f"\n>>> {fname}  ({len(bucket)} dòng)")
        for k, v in c.most_common():
            print(f"      {v:4}  {k}")


if __name__ == "__main__":
    main()
