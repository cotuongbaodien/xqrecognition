"""Compare GRID quality (cell occupancy, ignoring piece TYPE) of two board-seg
models against ground truth. FEN-exact match hides grid regressions when a
pre-existing piece-type error already marks a board wrong; this measures the
grid directly: how many cells differ in OCCUPANCY (piece present vs empty),
mirror-tolerant. Flags boards where the candidate grid is WORSE than baseline.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from boarddetection.pipeline import XiangqiRecognizer
from boarddetection.board_segmenter import BoardSegmenter

TEST_DIR = ROOT / "test"
GT = TEST_DIR / "ground_truth.txt"
OLD = "models/backups/board_seg_predeploy_2026-06-16.pt"
V5 = "models/backups/board_seg_v5.pt"


def parse_gt(p):
    g = {}
    for ln in open(p, encoding="utf-8"):
        if ":" in ln:
            k, v = ln.split(":", 1)
            g[k.strip()] = v.strip().split()[0]
    return g


def occ(fen):
    rows = []
    for row in fen.split("/"):
        cells = []
        for ch in row:
            cells += [0] * int(ch) if ch.isdigit() else [1]
        rows.append((cells + [0] * 9)[:9])
    while len(rows) < 10:
        rows.append([0] * 9)
    return rows[:10]


def occ_diff(a, b):
    ra, rb = occ(a), occ(b)
    return sum(1 for i in range(10) for j in range(9) if ra[i][j] != rb[i][j])


def mirror(fen):
    return "/".join("".join(str(x) for x in r[::-1])
                    for r in [[c for c in row] for row in occ(fen)])


def grid_err(det, gt):
    # occupancy mismatch vs GT, mirror-tolerant (min of direct / mirrored GT)
    g2 = "/".join("".join(str(x) for x in r[::-1]) for r in occ(gt))
    return min(occ_diff(det, gt), occ_diff(det, g2))


def run(rec, mp, imgs):
    rec.board_segmenter = BoardSegmenter(mp)
    return {i.stem: rec.recognize(str(i)).fen.split()[0] for i in imgs}


def main():
    gt = parse_gt(GT)
    imgs = sorted([f for f in TEST_DIR.iterdir()
                   if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
                  key=lambda p: int(p.stem) if p.stem.isdigit() else 1e9)
    rec = XiangqiRecognizer()
    print("running OLD..."); old = run(rec, OLD, imgs)
    print("running v5...");  v5 = run(rec, V5, imgs)

    worse, better, same = [], [], 0
    old_perfect = v5_perfect = 0
    for k, g in gt.items():
        eo, ev = grid_err(old.get(k, ""), g), grid_err(v5.get(k, ""), g)
        old_perfect += eo == 0
        v5_perfect += ev == 0
        if ev > eo:
            worse.append((k, eo, ev))
        elif ev < eo:
            better.append((k, eo, ev))
        else:
            same += 1
    print(f"\nGRID-perfect (occupancy exact, mirror-tol): OLD {old_perfect}/{len(gt)}  v5 {v5_perfect}/{len(gt)}")
    print(f"same={same}  better={len(better)}  worse={len(worse)}")
    print("\nGRID WORSE on v5 (img: old_err -> v5_err):")
    for k, eo, ev in sorted(worse, key=lambda t: int(t[0]) if t[0].isdigit() else 1e9):
        print(f"  {k:4s} {eo} -> {ev}")
    print("\nGRID BETTER on v5:")
    for k, eo, ev in sorted(better, key=lambda t: int(t[0]) if t[0].isdigit() else 1e9):
        print(f"  {k:4s} {eo} -> {ev}")


if __name__ == "__main__":
    main()
