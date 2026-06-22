"""Step 0 — GRID QC for synthetic-data generation.

Runs the board-segmentation model on every empty-board background, builds the
9x10 grid homography from the detected 4 corners, and draws the full lattice
(90 intersections + 4 corners + palace cells) onto each image so a human can
verify the grid is correct BEFORE any piece compositing. Wrong corners ->
wrong homography -> garbage synthetic labels, so this is a hard gate.

Also writes corners.json (the per-board 4 corners) which doubles as the
--corner-cache for synth_gen.py: boards whose grid looks wrong can be hand-
corrected (edit the quad) or removed here, once.

Usage:
    python scripts/viz_board_grid.py --src data/empty_boards \
        --out data/empty_boards/_grid_qc --cache data/empty_boards/corners.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from boarddetection.board_segmenter import BoardSegmenter  # noqa: E402
from boarddetection.settings import MODELS_DIR, GRID_COLS, GRID_ROWS  # noqa: E402

# Landmark grid cells (col,row) — mirror of item_detector._LANDMARK_GRID_POSITIONS
PALACE_CENTER = [(4, 1), (4, 8)]
PALACE_CONNER = [(3, 2), (5, 2), (3, 7), (5, 7)]
PALACE_BOTTOM = [(3, 0), (5, 0), (3, 9), (5, 9)]

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def homography(quad):
    """grid (col,row) -> image px. quad = (tl,tr,bl,br)."""
    src = np.float32([[0, 0], [8, 0], [0, 9], [8, 9]])
    dst = np.float32(list(quad))
    return cv2.getPerspectiveTransform(src, dst)


def project(H, col, row):
    p = H @ np.array([col, row, 1.0])
    return (p[0] / p[2], p[1] / p[2])


def draw_grid(img, H):
    """Draw 9x10 lattice + corners + palace cells onto a copy of img."""
    vis = img.copy()
    pts = {(c, r): project(H, c, r) for c in range(GRID_COLS) for r in range(GRID_ROWS)}

    def ipt(c, r):
        x, y = pts[(c, r)]
        return (int(round(x)), int(round(y)))

    # lattice lines (green)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 1):
            cv2.line(vis, ipt(c, r), ipt(c + 1, r), (0, 200, 0), 1, cv2.LINE_AA)
    for c in range(GRID_COLS):
        for r in range(GRID_ROWS - 1):
            cv2.line(vis, ipt(c, r), ipt(c, r + 1), (0, 200, 0), 1, cv2.LINE_AA)
    # intersections (yellow dots)
    for (c, r), (x, y) in pts.items():
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1, cv2.LINE_AA)
    # palace cells
    for c, r in PALACE_CENTER:
        cv2.circle(vis, ipt(c, r), 6, (255, 0, 255), 2, cv2.LINE_AA)   # magenta
    for c, r in PALACE_CONNER:
        cv2.circle(vis, ipt(c, r), 5, (255, 128, 0), 2, cv2.LINE_AA)   # orange
    for c, r in PALACE_BOTTOM:
        cv2.circle(vis, ipt(c, r), 5, (0, 128, 255), 2, cv2.LINE_AA)   # blue
    # 4 board corners (red, big) + label
    labels = [("TL", 0, 0), ("TR", 8, 0), ("BL", 0, 9), ("BR", 8, 9)]
    for name, c, r in labels:
        p = ipt(c, r)
        cv2.circle(vis, p, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(vis, name, (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/empty_boards")
    ap.add_argument("--out", default="data/empty_boards/_grid_qc")
    ap.add_argument("--cache", default="data/empty_boards/corners.json")
    ap.add_argument("--model", default=str(MODELS_DIR / "board_seg.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    src = PROJECT_ROOT / args.src
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    seg = BoardSegmenter(args.model)

    imgs = [p for p in src.rglob("*")
            if p.suffix.lower() in IMG_EXTS and "_grid_qc" not in p.parts]
    imgs.sort()
    print(f"QC {len(imgs)} empty boards with {args.model}")

    cache, ok, clipped, failed = {}, 0, 0, 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            failed += 1
            continue
        res = seg.detect(img, confidence=args.conf)
        rel = p.relative_to(src)
        if res is None or res.quad is None:
            failed += 1
            print(f"  FAIL (no board): {rel}")
            continue
        H = homography(res.quad)
        vis = draw_grid(img, H)
        flag = ""
        if res.any_clipped:
            clipped += 1
            flag = " [CLIPPED]"
            cv2.putText(vis, "CLIPPED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)
        ok += 1
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst.with_suffix(".jpg")), vis)
        cache[str(rel).replace("\\", "/")] = {
            "quad": [[float(x), float(y)] for x, y in res.quad],
            "palace_centers": [[float(x), float(y)] for x, y in res.palace_centers],
            "clipped": res.clipped,
            "source": "seg",
        }
        if flag:
            print(f"  ok{flag}: {rel}")

    (PROJECT_ROOT / args.cache).write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone: {ok} ok ({clipped} clipped), {failed} failed.")
    print(f"Overlays: {out}")
    print(f"Corners cache: {args.cache}")
    print("\nReview overlays. For any wrong grid: fix the 'quad' in corners.json "
          "(4 pts tl,tr,bl,br) or delete that entry to exclude the board.")


if __name__ == "__main__":
    main()
