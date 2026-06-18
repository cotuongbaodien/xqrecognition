"""Evaluate items models on the 224-image bench set (test/bench).
Same board_seg for all. Reports board-level FEN-exact (mirror-tolerant) and
piece-level error totals (missing / extra / wrong-class).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from boarddetection.pipeline import XiangqiRecognizer

BENCH = ROOT / "test/bench"
MODELS = {
    "old items.pt":   "boarddetection/models/items.pt",
    "v16 (+notok)":   "models/backups/items_v16_960.pt",
    "v16c (clean)":   "models/backups/items_v16c_960.pt",
}


def parse_gt(p):
    g = {}
    for ln in open(p, encoding="utf-8"):
        if ":" in ln and not ln.strip().startswith("#"):
            k, v = ln.split(":", 1)
            if v.strip():
                g[k.strip()] = v.strip()
    return g


def exp(f):
    rows = []
    for row in f.split()[0].split("/"):
        c = []
        for ch in row:
            c += ["."] * int(ch) if ch.isdigit() else [ch]
        rows.append((c + ["."] * 9)[:9])
    while len(rows) < 10:
        rows.append(["."] * 9)
    return rows[:10]


def mir(f):
    return "/".join("".join(r[::-1]) for r in exp(f))


def board_err(det, gt):
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
        if best is None or (miss + extra + wrong) < sum(best):
            best = (miss, extra, wrong)
    return best


def img_for(stem):
    for ext in (".png", ".jpg", ".jpeg"):
        p = BENCH / "images" / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None


def main():
    gt = parse_gt(BENCH / "ground_truth.txt")
    rec = XiangqiRecognizer()
    results = {}
    print(f"GT entries: {len(gt)}\n")
    for name, mp in MODELS.items():
        rec.item_detector.load_model(mp)
        miss = extra = wrong = exact = 0
        wrongs = []
        for stem, g in gt.items():
            ip = img_for(stem)
            if ip is None:
                continue
            det = rec.recognize(ip).fen.split()[0]
            m, e, w = board_err(det, g)
            miss += m; extra += e; wrong += w
            if m + e + w == 0:
                exact += 1
            else:
                wrongs.append(stem)
        results[name] = (exact, miss, extra, wrong, wrongs)
        print(f"done {name}")
    print(f"\n{'model':16s} | exact | MISS | EXTRA | WRONG | total")
    print("-" * 62)
    for name, (exact, miss, extra, wrong, _) in results.items():
        print(f"{name:16s} | {exact:3d}/{len(gt)} | {miss:4d} | {extra:5d} | {wrong:5d} | {miss+extra+wrong:5d}")


if __name__ == "__main__":
    main()
