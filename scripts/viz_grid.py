"""Render the recognized grid + pieces over each test image for visual review.

Draws the 9x10 grid lines, intersection dots, detected-piece cell markers, and
the predicted vs ground-truth FEN. Files are prefixed OK_ / WRONG_ (mirror-
tolerant) so the failures sort together for analysis.

Usage:
    python scripts/viz_grid.py [--out test/visual_v5]
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from boarddetection.pipeline import XiangqiRecognizer
from boarddetection.settings import GRID_COLS, GRID_ROWS

TEST_DIR = ROOT / "test"
GT_PATH = TEST_DIR / "ground_truth.txt"


def parse_gt(path):
    gt = {}
    for line in open(path, encoding="utf-8"):
        if ":" in line:
            k, v = line.split(":", 1)
            gt[k.strip()] = v.strip().split()[0]
    return gt


def expand_rows(fen):
    rows = []
    for row in fen.split("/"):
        cells = []
        for ch in row:
            cells += ["."] * int(ch) if ch.isdigit() else [ch]
        rows.append((cells + ["."] * 9)[:9])
    while len(rows) < 10:
        rows.append(["."] * 9)
    return rows[:10]


def mirror_fen(fen):
    return "/".join("".join(r[::-1]) for r in expand_rows(fen))


def matches(det, gt):
    return expand_rows(det) == expand_rows(gt) or expand_rows(det) == expand_rows(mirror_fen(gt))


def draw_grid(img, grid):
    pts = grid.points
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            p = tuple(np.round(pts[r, c]).astype(int))
            if c < GRID_COLS - 1:
                q = tuple(np.round(pts[r, c + 1]).astype(int))
                cv2.line(img, p, q, (0, 255, 0), 1)
            if r < GRID_ROWS - 1:
                q = tuple(np.round(pts[r + 1, c]).astype(int))
                cv2.line(img, p, q, (0, 255, 0), 1)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cv2.circle(img, tuple(np.round(pts[r, c]).astype(int)), 3, (0, 0, 255), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TEST_DIR / "visual_v5"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt = parse_gt(GT_PATH)
    images = sorted(
        [f for f in TEST_DIR.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 1e9,
    )
    rec = XiangqiRecognizer()
    n_ok = 0
    wrong = []
    for ip in images:
        img = cv2.imread(str(ip))
        res = rec.recognize(str(ip))
        det = res.fen.split()[0]
        ok = ip.stem in gt and matches(det, gt[ip.stem])
        n_ok += ok
        if ip.stem in gt and not ok:
            wrong.append(ip.stem)

        vis = img.copy()
        if res.grid is not None:
            draw_grid(vis, res.grid)
        # FEN overlay (top banner)
        banner = vis.shape[0] // 18
        cv2.rectangle(vis, (0, 0), (vis.shape[1], banner * 2 + 6), (0, 0, 0), -1)
        cv2.putText(vis, f"det: {det}", (5, banner), cv2.FONT_HERSHEY_SIMPLEX,
                    banner / 40, (0, 255, 0), 1, cv2.LINE_AA)
        gtxt = gt.get(ip.stem, "(no GT)")
        cv2.putText(vis, f"gt : {gtxt}", (5, banner * 2), cv2.FONT_HERSHEY_SIMPLEX,
                    banner / 40, (0, 200, 255), 1, cv2.LINE_AA)
        if res.errors:
            cv2.putText(vis, "ERR:" + ";".join(res.errors)[:60], (5, banner * 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, banner / 60, (0, 0, 255), 1, cv2.LINE_AA)

        tag = "OK" if ok else "WRONG"
        cv2.imwrite(str(out / f"{tag}_{ip.stem}.png"), vis)

    print(f"saved {len(images)} visuals to {out}")
    print(f"exact/mirror-OK: {n_ok}/{len(gt)}")
    print(f"WRONG ({len(wrong)}): {', '.join(wrong)}")


if __name__ == "__main__":
    main()
