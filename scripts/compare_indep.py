"""Cross-check missed pieces with an independent piece detector.

The deployed model (v12) was trained on our dataset, so it shares its
blind spots. An independent detector trained on a separate piece dataset
finds pieces v12 is blind to. For each test image we compare per-class
piece COUNTS: ground-truth vs v12 vs independent. Where v12 < GT but the
independent model reaches GT, that piece IS detectable -> a confirmed
recall gap to add to v12's training.

    python scripts/compare_indep.py --model runs/pieces/pieces_indep/weights/best.pt
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

# independent dataset class name -> our FEN symbol (case = color)
INDEP_TO_FEN = {
    "Black_Advisor": "a", "Black_Bishop": "b", "Black_Cannon": "c",
    "Black_King": "k", "Black_Knight": "n", "Black_Pawn": "p", "Black_Rook": "r",
    "Red_Advisor": "A", "Red_Bishop": "B", "Red_Cannon": "C",
    "Red_King": "K", "Red_Knight": "N", "Red_Pawn": "P", "Red_Rook": "R",
}
SYM_NAME = {"r": "xe", "n": "ma", "b": "tuong", "a": "si", "k": "tuong(soai)",
            "c": "phao", "p": "tot"}


def name(sym):
    return SYM_NAME[sym.lower()] + ("-do" if sym.isupper() else "-den")


def fen_counts(fen):
    c = Counter()
    for ch in fen.split()[0].replace("/", ""):
        if ch.isalpha():
            c[ch] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="independent piece model .pt")
    ap.add_argument("--gt", default="test/ground_truth.txt")
    ap.add_argument("--v12", default="test/output_v12_960/results.json")
    ap.add_argument("--imgs", default="test")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", default="indep_gap.csv")
    args = ap.parse_args()

    gt = {}
    for ln in (PROJECT_ROOT / args.gt).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            k, v = ln.split(":", 1)
            gt[int(k)] = v.strip()

    v12 = {}
    for x in json.load(open(PROJECT_ROOT / args.v12)):
        m = re.search(r"(\d+)", str(x.get("image") or x.get("file") or x.get("name")))
        if m:
            v12[int(m.group(1))] = x.get("fen") or ""

    model = YOLO(str(PROJECT_ROOT / args.model))
    names = model.names

    gaps = []   # (img, sym, gt_n, v12_n, indep_n)
    for n in sorted(gt):
        if n == 29 or n not in v12:
            continue
        img = PROJECT_ROOT / args.imgs / f"{n}.png"
        if not img.exists():
            img = PROJECT_ROOT / args.imgs / f"{n}.jpg"
        r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        indep = Counter()
        for b in r.boxes:
            sym = INDEP_TO_FEN.get(names[int(b.cls)])
            if sym:
                indep[sym] += 1
        gc, vc = fen_counts(gt[n]), fen_counts(v12[n])
        for sym in gc:
            # v12 misses this class AND independent recovers it
            if vc[sym] < gc[sym] and indep[sym] >= gc[sym]:
                gaps.append((n, sym, gc[sym], vc[sym], indep[sym]))

    import csv
    with (PROJECT_ROOT / args.out).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "piece", "gt_count", "v12_count", "indep_count"])
        for g in gaps:
            w.writerow([g[0], name(g[1]), g[2], g[3], g[4]])

    print(f"=== Confirmed recall gaps (v12 misses, independent recovers) ===")
    print(f"{len(gaps)} gaps tren {len(set(g[0] for g in gaps))} anh")
    by_piece = Counter(name(g[1]) for g in gaps)
    for k, v in by_piece.most_common():
        print(f"  {v:3}  {k}")
    print(f"CSV: {args.out}")
    print("\nChi tiet:")
    for img, sym, g, v, ind in gaps:
        print(f"  Anh {img}: {name(sym)}  GT={g} v12={v} indep={ind}")


if __name__ == "__main__":
    main()
