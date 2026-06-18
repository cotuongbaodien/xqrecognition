"""Piece-level error totals across the 86-image test set for several items
models (same board_seg). Finer than board-level FEN-exact: counts MISSING
(GT piece, detected empty), EXTRA (detected piece, GT empty), and WRONG-CLASS
(both occupied, different piece) summed over all boards. Mirror-tolerant
(per board, aligns to the GT orientation that minimizes total error).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from boarddetection.pipeline import XiangqiRecognizer

TEST = ROOT / "test"
MODELS = {
    "old items.pt": "boarddetection/models/items.pt",
    "v16 (+notok)": "models/backups/items_v16_960.pt",
    "v16c (clean)": "models/backups/items_v16c_960.pt",
}


def parse_gt(p):
    g = {}
    for ln in open(p, encoding="utf-8"):
        if ":" in ln:
            k, v = ln.split(":", 1); g[k.strip()] = v.strip().split()[0]
    return g


def exp(f):
    r = []
    for row in f.split("/"):
        c = []
        for ch in row:
            c += ["."] * int(ch) if ch.isdigit() else [ch]
        r.append((c + ["."] * 9)[:9])
    while len(r) < 10: r.append(["."] * 9)
    return r[:10]


def mir(f): return "/".join("".join(r[::-1]) for r in exp(f))


def board_err(det, gt):
    """min over gt/mirror: (missing, extra, wrong) plus board-exact flag."""
    best = None
    for g in (gt, mir(gt)):
        ra, rb = exp(det), exp(g)
        miss = extra = wrong = 0
        for i in range(10):
            for j in range(9):
                a, b = ra[i][j], rb[i][j]
                if a == "." and b != ".": miss += 1
                elif a != "." and b == ".": extra += 1
                elif a != "." and b != "." and a != b: wrong += 1
        if best is None or sum((miss, extra, wrong)) < sum(best):
            best = (miss, extra, wrong)
    return best


def run(rec, mp, imgs):
    rec.item_detector.load_model(mp)
    return {i.stem: rec.recognize(str(i)).fen.split()[0] for i in imgs}


def main():
    gt = parse_gt(TEST / "ground_truth.txt")
    imgs = sorted([f for f in TEST.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
                  key=lambda p: int(p.stem) if p.stem.isdigit() else 1e9)
    rec = XiangqiRecognizer()
    print(f"{'model':16s} | exact | MISS | EXTRA | WRONG | total")
    print("-" * 60)
    for name, mp in MODELS.items():
        det = run(rec, mp, imgs)
        miss = extra = wrong = exact = 0
        for k in gt:
            if k not in det: continue
            m, e, w = board_err(det[k], gt[k])
            miss += m; extra += e; wrong += w
            exact += (m + e + w) == 0
        tot = miss + extra + wrong
        print(f"{name:16s} | {exact:5d} | {miss:4d} | {extra:5d} | {wrong:5d} | {tot:5d}")


if __name__ == "__main__":
    main()
