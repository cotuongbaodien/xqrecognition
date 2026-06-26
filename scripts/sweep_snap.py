"""Sweep piece->cell snapping: Euclid vs perspective(H^-1), and contact ratio.

A piece is a 3D disc on a tilted plane: (1) the board plane is perspective ->
snap in canonical grid space via the inverse homography, not image space;
(2) the disc has height -> snap its BASE (contact point down the bbox), not the
bbox center. Sweeps both and reports straight(219)/skewed(24) exact-FEN.
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BENCH = os.path.join(ROOT, "test", "bench")
SKEW = {f"{n:03d}" for n in range(226, 250)}

import boarddetection.board_detector as bd          # noqa
import boarddetection.fen_generator as fg           # noqa
from boarddetection.pipeline import XiangqiRecognizer  # noqa


def parse_gt(p):
    g = {}
    for ln in open(p, encoding="utf-8"):
        if ":" in ln and ln.split(":", 1)[1].strip():
            g[ln.split(":")[0].strip()] = ln.split(":", 1)[1].strip().split()[0]
    return g


def exp(f):
    R = []
    for row in f.split("/"):
        c = []
        for ch in row:
            c += ["."] * int(ch) if ch.isdigit() else [ch]
        R.append((c + ["."] * 9)[:9])
    while len(R) < 10:
        R.append(["."] * 9)
    return R[:10]


def mir(f):
    return [r[::-1] for r in exp(f)]


def ok(det, g):
    return exp(det) == exp(g) or exp(det) == mir(g)


CONFIGS = [
    ("Euclid + center (goc)", False, 0.50),
    ("H-inv + center",        True,  0.50),
    ("H-inv + chan 0.62",     True,  0.62),
    ("H-inv + chan 0.70",     True,  0.70),
    ("H-inv + chan 0.78",     True,  0.78),
    ("H-inv + chan 0.85",     True,  0.85),
]


def main():
    gt = parse_gt(os.path.join(BENCH, "ground_truth.txt"))
    items = []
    for b, g in gt.items():
        ip = glob.glob(os.path.join(BENCH, "images", b + ".*"))
        if ip:
            items.append((b, ip[0], g))
    rec = XiangqiRecognizer()
    print(f"{'config':24s} | straight        | skewed       | total")
    print("-" * 66)
    for label, persp, ratio in CONFIGS:
        bd.USE_PERSPECTIVE_SNAP = persp
        fg.SNAP_CONTACT_RATIO = ratio
        se = st = sk = kt = 0
        for b, ip, g in items:
            det = rec.recognize(ip).fen.split()[0]
            good = ok(det, g)
            if b in SKEW:
                kt += 1; sk += good
            else:
                st += 1; se += good
        print(f"{label:24s} | {se:3d}/{st} ({100*se/st:4.1f}%) "
              f"| {sk:2d}/{kt} ({100*sk/kt:4.1f}%) | {se+sk}/{st+kt}")


if __name__ == "__main__":
    main()
