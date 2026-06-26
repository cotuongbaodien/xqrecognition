"""Synthesize PIECE-FULL board_seg training images from empty boards.

Composites real piece crops onto empty boards (reusing synth_gen's compositing),
then writes board_seg seg-labels (board quad + 2 palace quads) DERIVED from the
board homography — so the masks are geometrically exact and convention-consistent
(playing-grid area, true palace regions), not hand-drawn. This turns out-of-
distribution EMPTY boards into in-distribution PIECE-OCCLUDED boards for board_seg.

Output: data/board_seg_synth/train/{images,labels} (synth only — merged after).
"""
import json
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from synth_gen import (imread_u, imwrite_u, homography, project, cell_quad,   # noqa
                       composite_piece, load_pools, load_fens, FEN2ID, finalize)

BOARDS = os.path.join(ROOT, "data", "empty_boards")
CORNERS = os.path.join(ROOT, "data", "empty_boards", "corners.json")
CROPS = os.path.join(ROOT, "data", "piece_crops_all")
FENS = os.path.join(ROOT, "data", "fendata", "fendata", "labels.csv")
OUT = os.path.join(ROOT, "data", "board_seg_synth")
N_SYNTH = int(sys.argv[1]) if len(sys.argv) > 1 else 345

PALACE_TOP = [(3, 0), (5, 0), (5, 2), (3, 2)]
PALACE_BOT = [(3, 7), (5, 7), (5, 9), (3, 9)]


def norm(pts, W, H):
    return " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in pts)


def main():
    rng = random.Random(42)
    corners = json.load(open(CORNERS, encoding="utf-8"))
    boards = []
    for rel, info in corners.items():
        p = os.path.join(BOARDS, rel)
        if os.path.exists(p) and info.get("quad"):
            boards.append((p, info["quad"]))
    pools = load_pools(CROPS)
    fens = load_fens([FENS])
    print(f"boards={len(boards)} fens={len(fens)} "
          f"crops/class0 real={len(pools[0]['real'])} dh={len(pools[0]['dh'])}")

    def pick(cid):
        cats = [c for c in ("real", "dh", "dl") if pools[cid][c]]
        return rng.choice(pools[cid][rng.choice(cats)]) if cats else None

    di = os.path.join(OUT, "train", "images")
    dl = os.path.join(OUT, "train", "labels")
    os.makedirs(di, exist_ok=True)
    os.makedirs(dl, exist_ok=True)
    made = 0
    for k in range(N_SYNTH):
        path, quad = rng.choice(boards)
        board = imread_u(path)
        if board is None:
            continue
        img = board.copy()
        Himg, Wimg = img.shape[:2]
        H = homography(quad)
        fen = rng.choice(fens)
        occ = [(c, r) for r in range(10) for c in range(9) if fen[r][c]]
        occ.sort(key=lambda cr: project(H, cr[0], cr[1])[1])   # far -> near
        for (c, r) in occ:
            cid = FEN2ID.get(fen[r][c])
            if cid is None:
                continue
            crop = pick(cid)
            if crop is None:
                continue
            q = cell_quad(H, c, r, piece_height=0.20)
            composite_piece(img, crop, q, rng)
        img = finalize(img, rng)
        # ---- board_seg seg-label derived from homography ----
        tl, tr, bl, br = quad
        lines = ["0 " + norm([tl, tr, br, bl], Wimg, Himg)]            # board
        for pal in (PALACE_TOP, PALACE_BOT):                          # 2 palaces
            pts = [project(H, c, r) for (c, r) in pal]
            lines.append("1 " + norm(pts, Wimg, Himg))
        stem = f"bsynth_{k:05d}"
        imwrite_u(os.path.join(di, stem + ".jpg"), img)
        open(os.path.join(dl, stem + ".txt"), "w").write("\n".join(lines) + "\n")
        made += 1
        if made % 100 == 0:
            print(f"  {made}/{N_SYNTH}")
    print(f"\nDone: {made} synth board_seg images -> {OUT}")


if __name__ == "__main__":
    main()
