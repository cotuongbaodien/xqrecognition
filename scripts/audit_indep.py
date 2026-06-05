"""Find under-labeled images using an INDEPENDENT piece detector.

The deployed model shares blind spots with our labels (trained on them).
An independent detector flags pieces our dataset never labeled. For every
training image we run the independent model and report each detected piece
that has NO overlapping label box -> a candidate MISSING LABEL to fix
before retraining.

    python scripts/audit_indep.py --model runs/pieces/pieces_indep/weights/best.pt --data data/items_v12
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

INDEP_TO_FEN = {
    "Black_Advisor": "a", "Black_Bishop": "b", "Black_Cannon": "c",
    "Black_King": "k", "Black_Knight": "n", "Black_Pawn": "p", "Black_Rook": "r",
    "Red_Advisor": "A", "Red_Bishop": "B", "Red_Cannon": "C",
    "Red_King": "K", "Red_Knight": "N", "Red_Pawn": "P", "Red_Rook": "R",
}
SYM_NAME = {"r": "xe", "n": "ma", "b": "tuong", "a": "si", "k": "tuong(soai)",
            "c": "phao", "p": "tot"}
# our dataset class id (18-class scheme) -> FEN symbol; landmarks -> None
OUR_ID_TO_FEN = {0: "a", 1: "c", 2: "r", 3: "b", 4: "k", 5: "n", 6: "p",
                 7: None, 8: None, 9: None, 10: None,
                 11: "A", 12: "C", 13: "R", 14: "B", 15: "K", 16: "N", 17: "P"}
_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}


def name(sym):
    return SYM_NAME[sym.lower()] + ("-do" if sym.isupper() else "-den")


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def load_label_boxes(lbl, w, h):
    """Return list of (class_id, xyxy)."""
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cid = int(p[0])
        cx, cy, bw, bh = (float(x) for x in p[1:5])
        out.append((cid, [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]))
    return out


def src_name(fn):
    return re.split(r"\.rf\.", fn)[0]


def roboflow_name(src):
    parts = src.split("_")
    while parts and parts[-1].lower() in _EXTS:
        parts.pop()
    return "_".join(parts)


def cell_label(box, w, h):
    cx, cy = (box[0]+box[2])/2/w, (box[1]+box[3])/2/h
    return "abcdefghi"[min(8, max(0, int(cx*9)))] + str(min(9, max(0, int(cy*10))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/items_v12")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", default="indep_missing_labels.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = YOLO(str(PROJECT_ROOT / args.model))
    names = model.names
    # Scan EVERY split (train/valid/test) — valid/test labels have errors too.
    imgs = []
    for split in ("train", "valid", "test"):
        d = PROJECT_ROOT / args.data / split / "images"
        if d.exists():
            imgs += [(p, PROJECT_ROOT / args.data / split / "labels") for p in sorted(d.glob("*"))]
    if args.limit:
        imgs = imgs[:args.limit]

    seen = set()
    missing = []   # piece detected, no label there
    wrong = []     # piece detected, label there but DIFFERENT class
    for i, (img, lbl_dir) in enumerate(imgs):
        src = src_name(img.name)
        if src in seen:
            continue
        seen.add(src)
        r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        h, w = r.orig_shape
        labels = load_label_boxes(lbl_dir / (img.stem + ".txt"), w, h)
        for b in r.boxes:
            sym = INDEP_TO_FEN.get(names[int(b.cls)])
            if not sym:
                continue
            pbox = b.xyxy[0].tolist()
            # best-overlapping label
            best_lc, best_iou = None, 0.0
            for lc, lb in labels:
                v = iou(pbox, lb)
                if v > best_iou:
                    best_lc, best_iou = lc, v
            row = {
                "conf": round(float(b.conf), 2),
                "cell": cell_label(pbox, w, h),
                "roboflow_name": roboflow_name(src),
                "file": img.name,
            }
            if best_iou < args.iou:
                missing.append({**row, "piece": name(sym)})
            else:
                lsym = OUR_ID_TO_FEN.get(best_lc)
                if lsym and lsym != sym:   # label says a different piece -> mislabel
                    wrong.append({**row, "label_piece": name(lsym), "model_piece": name(sym)})
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(imgs)} | {len(seen)} src | {len(missing)} missing, {len(wrong)} wrong")

    missing.sort(key=lambda x: -x["conf"])
    wrong.sort(key=lambda x: -x["conf"])
    with (PROJECT_ROOT / args.out).open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=["piece", "conf", "cell", "roboflow_name", "file"])
        wri.writeheader()
        wri.writerows(missing)
    wrong_out = "indep_wrong_class.csv"
    with (PROJECT_ROOT / wrong_out).open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=["label_piece", "model_piece", "conf", "cell", "roboflow_name", "file"])
        wri.writeheader()
        wri.writerows(wrong)

    print(f"\n=== INDEPENDENT MODEL audit on {args.data} ===")
    print(f"Sources checked: {len(seen)}")
    print(f"\n[MISSING LABEL] {len(missing)} ca / {len(set(r['roboflow_name'] for r in missing))} anh -> {args.out}")
    for k, v in Counter(r["piece"] for r in missing).most_common():
        print(f"  {v:4}  {k}")
    print(f"\n[WRONG CLASS] {len(wrong)} ca / {len(set(r['roboflow_name'] for r in wrong))} anh -> {wrong_out}")
    for k, v in Counter(r["label_piece"] + " -> " + r["model_piece"] for r in wrong).most_common(15):
        print(f"  {v:4}  {k}")


if __name__ == "__main__":
    main()
