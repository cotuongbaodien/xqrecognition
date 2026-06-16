"""Side-by-side grid comparison OLD vs NEW board model for analysis.

For each image (default: those WRONG under the NEW model), renders the grid
from the OLD board model (left) and the deployed NEW model (right), with FEN +
per-side OK/WRONG vs ground truth. Lets you see at a glance whether the OLD
model localized a failing board better (a regression) or both fail (pre-
existing piece error).

Usage:
    python scripts/viz_compare.py                 # only NEW-wrong cases
    python scripts/viz_compare.py --all            # all 86
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from boarddetection.pipeline import XiangqiRecognizer
from boarddetection.board_segmenter import BoardSegmenter
from boarddetection.settings import GRID_COLS, GRID_ROWS

TEST_DIR = ROOT / "test"
GT_PATH = TEST_DIR / "ground_truth.txt"
OLD = "models/backups/board_seg_predeploy_2026-06-16.pt"   # pre-v5 board model
NEW = "boarddetection/models/board_seg.pt"                  # deployed v5_640


def parse_gt(p):
    g = {}
    for ln in open(p, encoding="utf-8"):
        if ":" in ln:
            k, v = ln.split(":", 1)
            g[k.strip()] = v.strip().split()[0]
    return g


def expand(fen):
    rows = []
    for row in fen.split("/"):
        cells = []
        for ch in row:
            cells += ["."] * int(ch) if ch.isdigit() else [ch]
        rows.append((cells + ["."] * 9)[:9])
    while len(rows) < 10:
        rows.append(["."] * 9)
    return rows[:10]


def mirror(fen):
    return "/".join("".join(r[::-1]) for r in expand(fen))


def ok(det, gt):
    return expand(det) == expand(gt) or expand(det) == expand(mirror(gt))


def draw_grid(img, grid, color=(0, 255, 0)):
    pts = grid.points
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            p = tuple(np.round(pts[r, c]).astype(int))
            if c < GRID_COLS - 1:
                cv2.line(img, p, tuple(np.round(pts[r, c + 1]).astype(int)), color, 2)
            if r < GRID_ROWS - 1:
                cv2.line(img, p, tuple(np.round(pts[r + 1, c]).astype(int)), color, 2)
            cv2.circle(img, p, 4, (0, 0, 255), -1)


def panel(img, grid, det, gt, label):
    v = img.copy()
    if grid is not None:
        draw_grid(v, grid)
    good = ok(det, gt) if gt else False
    h = v.shape[0]
    bh = max(46, h // 16)
    cv2.rectangle(v, (0, 0), (v.shape[1], bh * 2), (0, 0, 0), -1)
    col = (0, 255, 0) if good else (0, 0, 255)
    cv2.putText(v, f"{label}: {'OK' if good else 'WRONG'}", (8, int(bh * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 45, col, 2, cv2.LINE_AA)
    cv2.putText(v, det, (8, int(bh * 1.6)), cv2.FONT_HERSHEY_SIMPLEX,
                bh / 60, (255, 255, 255), 1, cv2.LINE_AA)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(TEST_DIR / "visual_compare"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    gt = parse_gt(GT_PATH)
    imgs = sorted([f for f in TEST_DIR.iterdir()
                   if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
                  key=lambda p: int(p.stem) if p.stem.isdigit() else 1e9)
    rec = XiangqiRecognizer()

    n = 0
    regress = []
    for ip in imgs:
        img = cv2.imread(str(ip))
        rec.board_segmenter = BoardSegmenter(NEW)
        rn = rec.recognize(str(ip)); dn = rn.fen.split()[0]
        new_ok = ip.stem in gt and ok(dn, gt[ip.stem])
        if new_ok and not args.all:
            continue
        rec.board_segmenter = BoardSegmenter(OLD)
        ro = rec.recognize(str(ip)); do = ro.fen.split()[0]
        old_ok = ip.stem in gt and ok(do, gt[ip.stem])
        if old_ok and not new_ok:
            regress.append(ip.stem)

        g = gt.get(ip.stem, "")
        left = panel(img, ro.grid, do, g, "OLD")
        right = panel(img, rn.grid, dn, g, "NEW")
        H = max(left.shape[0], right.shape[0])
        sep = np.full((H, 6, 3), 80, np.uint8)
        combo = cv2.hconcat([left, sep, right])
        tag = "REGRESS" if (old_ok and not new_ok) else ("BOTHWRONG" if not new_ok else "OK")
        cv2.imwrite(str(out / f"{tag}_{ip.stem}.png"), combo)
        n += 1

    print(f"saved {n} comparison images to {out}")
    print(f"REGRESS (old OK, new WRONG): {', '.join(regress) if regress else 'none'}")


if __name__ == "__main__":
    main()
