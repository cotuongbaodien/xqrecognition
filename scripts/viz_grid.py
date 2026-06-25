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
    ap.add_argument("--images-dir", default=str(TEST_DIR))
    ap.add_argument("--gt", default=str(GT_PATH))
    ap.add_argument("--out", default=str(TEST_DIR / "visual_v5"))
    ap.add_argument("--items", default=None, help="optional items model to swap in")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="piece conf (low => borderline pieces shown so you can "
                         "see what the 0.5 prod threshold would drop)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gt = parse_gt(args.gt)
    img_dir = Path(args.images_dir)
    images = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 1e9,
    )
    rec = XiangqiRecognizer()
    if args.items:
        rec.item_detector.load_model(args.items)
    n_ok = 0
    wrong = []
    plog = []   # per-piece conf + name log (text)
    for ip in images:
        img = cv2.imread(str(ip))
        res = rec.recognize(str(ip), piece_confidence=args.conf)
        det = res.fen.split()[0]
        ok = ip.stem in gt and matches(det, gt[ip.stem])
        n_ok += ok
        if ip.stem in gt and not ok:
            wrong.append(ip.stem)

        tag0 = "OK" if ok else "WRONG"
        plog.append(f"\n=== {tag0}_{ip.stem}  ({len(res.pieces)} quan, "
                    f"conf>={args.conf}) ===")
        for p in sorted(res.pieces, key=lambda x: x.confidence):
            flag = "  <== conf thap" if p.confidence < 0.5 else ""
            plog.append(f"  conf={p.confidence:.3f}  {p.class_name} "
                        f"({p.fen_symbol}){flag}")

        vis = img.copy()
        if res.grid is not None:
            draw_grid(vis, res.grid)
        # ---- map detections to grid cells, compare vs GT (4 orientations) ----
        det_cell = {}
        if res.grid is not None:
            for p in res.pieces:
                r, c = res.grid.get_nearest_cell(p.center[0], p.center[1])
                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                    det_cell[(r, c)] = p
        gt_rows = expand_rows(gt[ip.stem]) if ip.stem in gt else None
        wrong_cells, miss_cells, gtt = set(), {}, None
        if gt_rows and res.grid is not None:
            best = -1
            for hm in (False, True):
                for vf in (False, True):
                    g = [row[::-1] for row in gt_rows] if hm else [list(r) for r in gt_rows]
                    if vf:
                        g = g[::-1]
                    sc = sum(1 for (r, c), p in det_cell.items()
                             if g[r][c] == (p.fen_symbol or "?"))
                    if sc > best:
                        best, gtt = sc, g
            for (r, c), p in det_cell.items():
                if gtt[r][c] != (p.fen_symbol or "?"):
                    wrong_cells.add((r, c))          # wrong class OR extra (GT empty)
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    if gtt[r][c] != "." and (r, c) not in det_cell:
                        miss_cells[(r, c)] = gtt[r][c]    # GT has piece, model missed

        # label sits ABOVE the token so the piece stays visible
        pts = res.grid.points if res.grid is not None else None
        pitch = int(abs(pts[1, 0][1] - pts[0, 0][1])) if pts is not None else 26
        off = max(13, int(pitch * 0.45))
        PINK = (180, 105, 255)

        def label(x, y, txt, col, fs=0.45):
            cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)

        for (r, c), p in det_cell.items():
            cx, cy = int(p.center[0]), int(p.center[1])
            sym = p.fen_symbol or "?"
            col = PINK if (r, c) in wrong_cells else (
                (0, 0, 220) if sym.isupper() else (10, 10, 10))  # red / black
            label(cx - 16, cy - off, f"{sym} {p.confidence:.2f}", col)
        # missing pieces: pink ring + expected GT symbol at the empty cell
        for (r, c), sym in miss_cells.items():
            x, y = int(pts[r, c][0]), int(pts[r, c][1])
            cv2.circle(vis, (x, y), max(9, off), PINK, 2, cv2.LINE_AA)
            label(x - 7, y + 5, sym, PINK, 0.6)
        # FEN info on a SEPARATE strip above the board (never covers it)
        gtxt = gt.get(ip.stem, "(no GT)")
        lines = [(f"det: {det}", (0, 255, 0)),
                 (f"gt : {gtxt}", (0, 200, 255))]
        if res.errors:
            lines.append(("ERR: " + ";".join(res.errors), (0, 0, 255)))
        W = vis.shape[1]

        def fit_scale(txt):
            for s in (0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.22):
                tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, s, 1)[0][0]
                if tw <= W - 10:
                    return s
            return 0.22
        scale = min(fit_scale(t) for t, _ in lines)
        th = cv2.getTextSize("Ag/0", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][1]
        lh = th + 9
        strip = np.zeros((lh * len(lines) + 8, W, 3), np.uint8)
        for i, (txt, col) in enumerate(lines):
            cv2.putText(strip, txt, (5, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, col, 1, cv2.LINE_AA)
        vis = cv2.vconcat([strip, vis])

        tag = "OK" if ok else "WRONG"
        cv2.imwrite(str(out / f"{tag}_{ip.stem}.png"), vis)

    (out / "_piece_conf_log.txt").write_text("\n".join(plog), encoding="utf-8")
    print(f"saved {len(images)} visuals to {out}")
    print(f"piece conf+name log -> {out / '_piece_conf_log.txt'}")
    print(f"exact/mirror-OK: {n_ok}/{len(gt)}")
    print(f"WRONG ({len(wrong)}): {', '.join(wrong)}")


if __name__ == "__main__":
    main()
