"""Prototype: simulate segmentation-based board localization.

Manually specify the 4 board-grid corners (what a perfect segmentation
model + perspective extraction would give), then run the EXISTING
grid + piece-snap + FEN pipeline. Tests whether good board localization
alone fixes the tilted-board failures (11, 13, 15) WITHOUT touching
piece detection.

Usage: python scripts/proto_seg_grid.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from boarddetection.item_detector import ItemDetector
from boarddetection.piece_detector import PieceDetector
from boarddetection.fen_generator import FENGenerator
from boarddetection.settings import ITEMS_MODEL

# Manual 4-corner grid annotations (col,row → image x,y).
# These simulate the output of a board-segmentation model: the 4 corners
# of the 9x10 intersection grid (NOT the wooden frame). Estimated by eye.
# Orientation assignment (which image corner is grid (0,0)) is arbitrary;
# the pipeline's orientation detection flips vertically by piece color.
MANUAL_CORNERS = {
    "11.jpg": {(0, 0): (245, 300), (8, 0): (528, 287),
               (0, 9): (168, 655), (8, 9): (562, 638)},
    "13.jpg": {(0, 0): (45, 465),  (8, 0): (395, 448),
               (0, 9): (120, 955), (8, 9): (540, 875)},
    "15.jpg": {(0, 0): (33, 290),  (8, 0): (427, 290),
               (0, 9): (30, 725),  (8, 9): (428, 725)},
}

GROUND_TRUTH = {
    "11.jpg": "3rka1r1/4a4/1c2b1n1c/p3pNR1p/1nb3p2/6P2/P3P3P/2C1C4/9/RNBAKAB2",
    "13.jpg": "2bak3r/4a4/3cb1nc1/2pnN1p1p/p5Pr1/2P1R4/P3P3P/2N1BC2C/4A4/2BAK3R",
    "15.jpg": "2baka3/9/4c3b/p7p/2p1P1p2/9/P1P3n1P/4B1N2/1r2Ac1C1/RN2KAB2",
}


def main():
    det = ItemDetector(str(ITEMS_MODEL))
    pdet = PieceDetector()
    fgen = FENGenerator()

    for fname, corners in MANUAL_CORNERS.items():
        img = cv2.imread(f"test/{fname}")
        result = det.detect(img, confidence=0.3)
        pieces = pdet.non_max_suppression(result.pieces, iou_threshold=0.35)
        if len(pieces) > 32:
            pieces = sorted(pieces, key=lambda p: p.confidence, reverse=True)[:32]

        corner_corrs = [(c, r, x, y) for (c, r), (x, y) in corners.items()]
        grid = ItemDetector._grid_from_4_corners(corner_corrs, image_shape=img.shape[:2])
        if grid is None:
            print(f"{fname}: grid build FAILED")
            continue

        board_state = fgen.map_pieces_to_grid(pieces, grid)
        orient = fgen.detect_board_orientation(board_state)
        if orient == "flipped":
            board_state = fgen.flip_board(board_state)
        fen = fgen.generate_fen(board_state).split(" ")[0]

        gt = GROUND_TRUTH[fname]
        rows_diff = sum(1 for a, b in zip(fen.split("/"), gt.split("/")) if a != b)
        status = "EXACT" if fen == gt else f"{rows_diff}/10 rows differ"
        print(f"\n{fname}: {status}")
        print(f"  detected: {fen}")
        print(f"  truth:    {gt}")


if __name__ == "__main__":
    main()
