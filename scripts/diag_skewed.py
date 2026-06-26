"""Diagnose WHERE board_seg fails on the skewed boards: mask -> quad -> grid."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boarddetection.board_segmenter import BoardSegmenter      # noqa
from boarddetection.pipeline import XiangqiRecognizer          # noqa
from boarddetection.settings import MODELS_DIR                 # noqa

SK = os.path.join(ROOT, "test", "bench_skewed", "images")
OUT = os.path.join(ROOT, "runs", "diag_skewed")
os.makedirs(OUT, exist_ok=True)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def main():
    seg = BoardSegmenter(str(MODELS_DIR / "board_seg.pt"))
    rec = XiangqiRecognizer()
    for f in sorted(os.listdir(SK)):
        ip = os.path.join(SK, f)
        im = imread_u(ip)
        if im is None:
            continue
        stem = os.path.splitext(f)[0]
        # --- board_seg raw ---
        try:
            r = seg.detect(im, confidence=0.25)
        except Exception as e:
            print(f"{stem}: SEG EXCEPTION {e}")
            r = None
        vis = im.copy()
        line = f"{stem}: "
        if r is None or r.quad is None:
            line += "QUAD=None (mask/board khong detect duoc) "
        else:
            q = [tuple(map(int, p)) for p in r.quad]   # tl,tr,bl,br
            for i, (nm, p) in enumerate(zip(["tl", "tr", "bl", "br"], q)):
                cv2.circle(vis, p, 10, (0, 0, 255), -1)
                cv2.putText(vis, nm, (p[0] + 6, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
            cv2.line(vis, q[0], q[1], (0, 255, 0), 2)   # top
            cv2.line(vis, q[0], q[2], (255, 0, 0), 2)   # left
            cv2.line(vis, q[1], q[3], (0, 255, 0), 2)
            cv2.line(vis, q[2], q[3], (0, 255, 0), 2)
            line += (f"quad OK | clipped={r.clipped} | "
                     f"palace_centers={len(r.palace_centers)} ")
        # --- full pipeline grid ---
        res = rec.recognize(ip)
        grid = res.grid
        if grid is not None:
            for rr in range(grid.points.shape[0]):
                for cc in range(grid.points.shape[1]):
                    x, y = grid.points[rr, cc]
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)
            line += "| pipeline GRID=built "
        else:
            line += "| pipeline GRID=None "
        line += f"| errors={res.errors}"
        print(line)
        cv2.imencode(".png", vis)[1].tofile(os.path.join(OUT, f"diag_{stem}.png"))
    print(f"\n-> {OUT}/diag_*.png")


if __name__ == "__main__":
    main()
