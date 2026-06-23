"""Overlay the production grid + board-seg quad + palace quads on bench boards.
Lets us SEE whether the failure on a broken board is a bad outer quad (grid
skewed) and whether the palace masks are good enough to anchor the interior.

Usage: python scripts/grid_debug.py 047 061 042 024 011 009
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from boarddetection.pipeline import XiangqiRecognizer
from boarddetection.board_segmenter import BoardSegmenter
from boarddetection.settings import MODELS_DIR, GRID_COLS, GRID_ROWS

BENCH = ROOT / "test/bench"
OUT = BENCH / "_grid_debug"
OUT.mkdir(exist_ok=True)


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def img_for(s):
    for e in (".jpg", ".png", ".jpeg"):
        p = BENCH / "images" / f"{s}{e}"
        if p.exists():
            return p


def main():
    ids = sys.argv[1:] or ["047", "061", "042", "024", "011", "009"]
    rec = XiangqiRecognizer()
    seg = BoardSegmenter(str(MODELS_DIR / "board_seg.pt"))
    for s in ids:
        p = img_for(s)
        if not p:
            print(f"{s}: no image"); continue
        im = imread_u(p)
        res = rec.recognize(str(p))
        sd = seg.detect(im)
        vis = im.copy()
        # board-seg quad (outer 4 corners) — RED
        if sd is not None:
            q = np.array(sd.quad, np.int32)
            cv2.polylines(vis, [q[[0, 1, 3, 2]]], True, (0, 0, 255), 3)
            for i, pt in enumerate(sd.quad):
                cv2.circle(vis, tuple(map(int, pt)), 8, (0, 0, 255), -1)
            # palace quads — YELLOW (the interior anchors going unused)
            for pq in sd.palace_quads:
                cv2.polylines(vis, [np.array(pq, np.int32)], True,
                              (0, 255, 255), 2)
            for pc in sd.palace_centers:
                cv2.circle(vis, tuple(map(int, pc)), 6, (0, 255, 255), -1)
        # final projected 9x10 grid — GREEN
        g = res.grid
        npal = len(sd.palace_quads) if sd else 0
        if g is not None:
            gp = g.points
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    x, y = int(gp[r, c, 0]), int(gp[r, c, 1])
                    if c < GRID_COLS - 1:
                        x2, y2 = int(gp[r, c + 1, 0]), int(gp[r, c + 1, 1])
                        cv2.line(vis, (x, y), (x2, y2), (0, 200, 0), 1)
                    if r < GRID_ROWS - 1:
                        x2, y2 = int(gp[r + 1, c, 0]), int(gp[r + 1, c, 1])
                        cv2.line(vis, (x, y), (x2, y2), (0, 200, 0), 1)
        cv2.imwrite(str(OUT / f"{s}.jpg"), vis)
        print(f"{s}: grid={'OK' if g else 'FAIL'} palaces={npal} "
              f"clipped={sd.any_clipped if sd else '?'} -> _grid_debug/{s}.jpg")


if __name__ == "__main__":
    main()
