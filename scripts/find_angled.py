"""Scan the items/piece dataset, run board_seg on each distinct-source image,
score how ANGLED (perspective trapezoid + in-plane rotation) the detected board
quad is, and copy the top-N most-angled images out for board_seg re-annotation.
"""
import sys, glob, os, re, math, shutil
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from boarddetection.board_segmenter import BoardSegmenter

POOL_GLOBS = [
    "data/items_v16/train/images/*", "data/items_v16/valid/images/*", "data/items_v16/test/images/*",
]
OUT = ROOT / "data/angled_pool"
TOP_N = 120


def base(f):
    n = re.split(r"\.rf\.", os.path.basename(f))[0]
    while True:
        n2 = re.sub(r"_(png|jpg|jpeg)$", "", n, flags=re.I)
        if n2 == n: break
        n = n2
    return n


def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])


def angled_score(quad):
    tl, tr, bl, br = quad
    top, bot = dist(tl, tr), dist(bl, br)
    left, right = dist(tl, bl), dist(tr, br)
    # perspective: opposite edges differ in length (trapezoid)
    persp = abs(top-bot)/max(top, bot, 1) + abs(left-right)/max(left, right, 1)
    # in-plane rotation of the top edge away from horizontal (0..90)
    a = abs(math.degrees(math.atan2(tr[1]-tl[1], tr[0]-tl[0])))
    rot = min(a, abs(180-a)) / 90.0
    return persp + 0.5*rot, persp, rot


def main():
    seg = BoardSegmenter("boarddetection/models/board_seg.pt")
    seen = {}
    for g in POOL_GLOBS:
        for f in glob.glob(str(ROOT/g)):
            seen.setdefault(base(f), f)
    print(f"distinct sources: {len(seen)}")
    scored = []
    for i, (b, f) in enumerate(seen.items()):
        img = cv2.imread(f)
        if img is None: continue
        r = seg.detect(img)
        if r is None: continue
        s, persp, rot = angled_score(r.quad)
        scored.append((s, persp, rot, f))
        if (i+1) % 300 == 0: print(f"  {i+1}/{len(seen)}")
    scored.sort(reverse=True)
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rep = open(OUT/"angled_ranked.csv", "w", encoding="utf-8")
    rep.write("rank,score,persp,rot,file\n")
    for i, (s, persp, rot, f) in enumerate(scored[:TOP_N], 1):
        ext = os.path.splitext(f)[1]
        shutil.copy(f, OUT/f"ang_{i:03d}{ext}")
        rep.write(f"{i},{s:.3f},{persp:.3f},{rot:.3f},{os.path.basename(f)}\n")
    rep.close()
    print(f"copied top {min(TOP_N,len(scored))} angled -> {OUT}")
    print(f"score range: {scored[0][0]:.2f} (most) .. {scored[min(TOP_N,len(scored))-1][0]:.2f}")


if __name__ == "__main__":
    main()
