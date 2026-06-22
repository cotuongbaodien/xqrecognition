"""Synthetic Xiangqi training-image generator.

Composites piece crops (digital skins + harvested real crops) onto empty-board
photos at the exact grid intersections (via the board homography), following
real FEN positions, and writes perfect YOLO 18-class labels (pieces + the 4
landmark classes). See plan: docs/SYNTHETIC_DATA_PLAN / plans file.

Pipeline per image: pick board -> H from its QC'd corners -> pick FEN -> for
each occupied cell paste a perspective-warped piece (feather+shadow+jitter) ->
emit label from the same cell footprint -> emit landmark labels -> finalize
(JPEG). Output is a flat or split YOLO dataset ready for merge/split/train.

Usage:
    python scripts/synth_gen.py --boards data/empty_boards \
        --corners data/empty_boards/corners.json \
        --png-root download/BO_COTUONG_CLEAN/PNG_hires \
        --real-root data/piece_crops_real \
        --fens data/fendata/fendata/labels.csv \
        --out data/items_synth_v1 --n 1000 --seed 42 --layout split
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from boarddetection.settings import ITEM_CLASSES, ITEM_CLASS_NAMES  # noqa: E402


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """Unicode-safe imread (cv2.imread fails on non-ASCII paths on Windows).
    Skin folders use Chinese names → MUST use this, not cv2.imread."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_u(path, img):
    """Unicode-safe imwrite."""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok

# fen symbol -> class id  (pieces only)
FEN2ID = {fen: cid for cid, (_, fen) in ITEM_CLASSES.items() if fen}
# class name -> id
NAME2ID = {name: cid for cid, (name, _) in ITEM_CLASSES.items()}
# piece-letter (k,a,b,n,r,c,p) + color -> class id
DIGI2ID = {}
for cid, (name, fen) in ITEM_CLASSES.items():
    if fen:
        DIGI2ID[("b" if fen.islower() else "r", fen.lower())] = cid

LANDMARKS = {
    "board-conner":  [(0, 0), (8, 0), (0, 9), (8, 9)],
    "palace-bottom": [(3, 0), (5, 0), (3, 9), (5, 9)],
    "palace-conner": [(3, 2), (5, 2), (3, 7), (5, 7)],
    "palace-center": [(4, 1), (4, 8)],
}


# --------------------------------------------------------------------------- #
# assets
# --------------------------------------------------------------------------- #
def load_crops(png_roots, real_root):
    """{class_id: [rgba, ...]} from digital skins + real harvested crops.
    png_roots: list of dirs each holding <skin>/{b|r}{k a b n r c p}.png."""
    pool = {cid: [] for cid in FEN2ID.values()}
    if isinstance(png_roots, str):
        png_roots = [png_roots]
    for png_root in png_roots:
        for f in glob.glob(f"{png_root}/*/*.png"):
            base = os.path.splitext(os.path.basename(f))[0].lower()
            if len(base) != 2 or base[0] not in "br" or base[1] not in "kabnrcp":
                continue
            cid = DIGI2ID.get((base[0], base[1]))
            if cid is None:
                continue
            im = imread_u(f, cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                pool[cid].append(im)
    # real crops: <class_name>/*.png
    for cid, name in [(c, ITEM_CLASSES[c][0]) for c in pool]:
        for f in glob.glob(f"{real_root}/{name}/*.png"):
            im = imread_u(f, cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                pool[cid].append(im)
    return pool


def load_pools(crops_root):
    """Consolidated lib → {cid: {'real':[], 'dh':[], 'dl':[]}} bucketed by the
    category prefix of each crop filename (real_ / dh### / dl###)."""
    pools = {cid: {"real": [], "dh": [], "dl": []} for cid in FEN2ID.values()}
    for cid in pools:
        name = ITEM_CLASSES[cid][0]
        for f in glob.glob(f"{crops_root}/{name}/*.png"):
            b = os.path.basename(f)
            cat = ("real" if b.startswith("real") else
                   "dh" if b.startswith("dh") else
                   "dl" if b.startswith("dl") else None)
            if cat is None:
                continue
            im = imread_u(f, cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim == 3 and im.shape[2] == 4:
                pools[cid][cat].append(im)
    return pools


def load_board_types(manifest_path):
    """{board_number_str: 'real'|'digital'} from manifest.csv."""
    t = {}
    if os.path.exists(manifest_path):
        for r in csv.DictReader(open(manifest_path, encoding="utf-8")):
            t[r["num"]] = r["type"]
    return t


def load_fens(paths):
    """Return list of 10x9 boards (symbol or None). Hardened for digit runs."""
    boards = []
    seen = set()
    for path in paths:
        path = str(path)
        rows_iter = []
        if path.endswith(".csv"):
            with open(path, encoding="utf-8") as fp:
                for row in csv.DictReader(fp):
                    rows_iter.append(row.get("fen", ""))
        else:  # "stem: fen ..." lines OR plain "fen" per line
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rows_iter.append(line.split(":", 1)[1] if ":" in line else line)
        for fen in rows_iter:
            fen = fen.strip().split()[0] if fen.strip() else ""
            if not fen or fen in seen:
                continue
            b = parse_fen(fen)
            if b is not None:
                seen.add(fen)
                boards.append(b)
    return boards


def parse_fen(fen):
    rows = fen.split("/")
    if len(rows) != 10:
        return None
    board = []
    for rs in rows:
        cells, i = [], 0
        while i < len(rs):
            ch = rs[i]
            if ch.isdigit():
                j = i
                while j < len(rs) and rs[j].isdigit():
                    j += 1
                cells.extend([None] * int(rs[i:j]))
                i = j
            else:
                cells.append(ch)
                i += 1
        if len(cells) != 9:
            return None
        board.append(cells)
    return board


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def homography(quad):
    src = np.float32([[0, 0], [8, 0], [0, 9], [8, 9]])
    dst = np.float32(quad)
    return cv2.getPerspectiveTransform(src, dst)


def project(H, col, row):
    p = H @ np.array([col, row, 1.0])
    return p[0] / p[2], p[1] / p[2]


def cell_quad(H, col, row, half=0.45, piece_height=0.0):
    """Footprint quad (tl,tr,bl,br) for cell. piece_height>0 lifts the TOP edge
    up in image-space by that fraction of the cell's vertical extent — mimics the
    3D height of a real piece (disc body above the intersection) so the warped
    piece + its bbox are TALLER, matching real labels (h/w~1.14, not flat 0.93)."""
    g = np.float32([[col - half, row - half, 1], [col + half, row - half, 1],
                    [col - half, row + half, 1], [col + half, row + half, 1]]).T
    p = H @ g
    q = (p[:2] / p[2]).T.astype(np.float32)   # tl,tr,bl,br
    if piece_height > 0:
        celly = 0.5 * (np.linalg.norm(q[2] - q[0]) + np.linalg.norm(q[3] - q[1]))
        dy = piece_height * celly
        q[0, 1] -= dy   # lift tl up
        q[1, 1] -= dy   # lift tr up
    return q


def quad_aabb_label(quad, W, H_img, cid):
    xs, ys = quad[:, 0], quad[:, 1]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    cx = max(0, min(1, (x1 + x2) / 2 / W))
    cy = max(0, min(1, (y1 + y2) / 2 / H_img))
    w = max(0, min(1, (x2 - x1) / W))
    h = max(0, min(1, (y2 - y1) / H_img))
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# --------------------------------------------------------------------------- #
# compositing + realism
# --------------------------------------------------------------------------- #
def apply_realism(rgba, rng, base_region):
    h, w = rgba.shape[:2]
    # rotation + scale jitter
    ang = rng.uniform(-4, 4)
    sc = rng.uniform(0.92, 1.05)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    rgba = cv2.warpAffine(rgba, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # brightness/hue match toward the board region it will land on
    bgr = rgba[:, :, :3].astype(np.float32)
    if base_region is not None and base_region.size:
        tgt = float(np.mean(cv2.cvtColor(base_region, cv2.COLOR_BGR2GRAY)))
        cur = float(np.mean(bgr)) + 1e-3
        f = np.clip(0.5 + 0.5 * (tgt / cur), 0.8, 1.2)  # gentle pull
        bgr = np.clip(bgr * f, 0, 255)
    out = rgba.copy()
    out[:, :, :3] = bgr.astype(np.uint8)
    return out   # blend (feather vs hard) decided per-piece in composite_piece


def composite_piece(base, rgba, quad, rng):
    Himg, Wimg = base.shape[:2]
    xs, ys = quad[:, 0], quad[:, 1]
    rx1, ry1 = max(0, int(xs.min())), max(0, int(ys.min()))
    rx2, ry2 = min(Wimg, int(xs.max())), min(Himg, int(ys.max()))
    region = base[ry1:ry2, rx1:rx2] if rx2 > rx1 and ry2 > ry1 else None
    rgba = apply_realism(rgba, rng, region)
    ph, pw = rgba.shape[:2]
    src = np.float32([[0, 0], [pw, 0], [0, ph], [pw, ph]])
    M = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(rgba, M, (Wimg, Himg), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    wb = warped[:, :, :3].astype(np.float32)
    a = warped[:, :, 3].astype(np.float32)
    # random blend mode per piece so the model can't key on one paste artifact
    mode = rng.random()
    if mode < 0.4:          # hard edge
        a = np.where(a >= 128, 255.0, 0.0)
    elif mode < 0.8:        # feather, random sigma
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.5, 2.5))
    # else: leave raw warped alpha (mild anti-alias)
    wa = (a[:, :, None] / 255.0)
    # drop shadow: offset, blurred, dark — composite UNDER the piece
    sh = warped[:, :, 3].astype(np.float32)
    Mt = np.float32([[1, 0, 5], [0, 1, 6]])
    sh = cv2.warpAffine(sh, Mt, (Wimg, Himg))
    sh = cv2.GaussianBlur(sh, (0, 0), 6)
    sa = (sh[:, :, None] / 255.0) * 0.35 * (1 - wa)
    base[:] = (base.astype(np.float32) * (1 - sa)).astype(np.uint8)
    base[:] = (wb * wa + base.astype(np.float32) * (1 - wa)).astype(np.uint8)


def finalize(img, rng):
    q = int(rng.uniform(78, 93))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default="data/empty_boards")
    ap.add_argument("--corners", default="data/empty_boards/corners.json")
    ap.add_argument("--crops-root", default="data/piece_crops_all",
                    help="consolidated ASCII crop library <class_name>/*.png "
                         "(digital+real merged). Preferred. Set '' to use "
                         "--png-root/--real-root instead.")
    ap.add_argument("--png-root", action="append",
                    help="digital skin dir(s); repeatable. "
                         "default: PNG_hires + piece_svg_png")
    ap.add_argument("--real-root", default="data/piece_crops_real")
    ap.add_argument("--fens", action="append", required=True)
    ap.add_argument("--out", default="data/items_synth_v1")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layout", choices=["flat", "split"], default="split")
    ap.add_argument("--digital-frac", type=float, default=0.5,
                    help="prob of using a digital crop vs real (per piece)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nprng = np.random.RandomState(args.seed)

    btypes = load_board_types(str(PROJECT_ROOT / args.boards / "manifest.csv"))
    corners = json.load(open(PROJECT_ROOT / args.corners, encoding="utf-8"))
    real_boards, digi_boards = [], []
    for rel, info in corners.items():
        p = PROJECT_ROOT / args.boards / rel
        if not p.exists():
            continue
        num = os.path.splitext(os.path.basename(rel))[0].lstrip("0") or "0"
        bt = btypes.get(num, "real")
        (real_boards if bt == "real" else digi_boards).append((str(p), info["quad"]))
    if not real_boards and not digi_boards:
        sys.exit("No boards with corners found.")

    pools = load_pools(str(PROJECT_ROOT / args.crops_root))
    fens = load_fens([PROJECT_ROOT / f if not os.path.isabs(f) else f
                      for f in args.fens])
    p0 = pools[0]
    print(f"boards: real={len(real_boards)} digital={len(digi_boards)} | "
          f"fens={len(fens)} | crops/class real={len(p0['real'])} "
          f"dh={len(p0['dh'])} dl={len(p0['dl'])}")
    if not fens:
        sys.exit("No FENs loaded.")

    out = PROJECT_ROOT / args.out
    if out.exists():
        import shutil
        shutil.rmtree(out)
    sub = "train/" if args.layout == "split" else ""
    img_dir = out / f"{sub}images"
    lbl_dir = out / f"{sub}labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # crop-source weights per board type (real boards favor real 3D crops;
    # digital boards favor clean digital; lowres only on digital, capped)
    WEIGHTS = {"real":    {"real": 0.8, "dh": 0.2, "dl": 0.0},
               "digital": {"real": 0.0, "dh": 0.9, "dl": 0.1}}

    def pick_crop(cid, btype):
        w = WEIGHTS[btype]
        cats = [c for c in ("real", "dh", "dl") if pools[cid][c] and w[c] > 0]
        if not cats:  # fallback: any non-empty category
            cats = [c for c in ("real", "dh", "dl") if pools[cid][c]]
            if not cats:
                return None
            cat = rng.choice(cats)
        else:
            cat = rng.choices(cats, weights=[w[c] for c in cats])[0]
        return rng.choice(pools[cid][cat])

    made = 0
    for k in range(args.n):
        # 70% real board, 30% digital (fallback if a list is empty)
        use_real = rng.random() < 0.70
        pool_b = real_boards if (use_real and real_boards) or not digi_boards else digi_boards
        btype = "real" if pool_b is real_boards else "digital"
        board_path, quad = rng.choice(pool_b)
        board = imread_u(board_path)
        if board is None:
            continue
        img = board.copy()
        Himg, Wimg = img.shape[:2]
        H = homography(quad)
        fen = rng.choice(fens)

        occ = [(c, r) for r in range(10) for c in range(9) if fen[r][c]]
        occ.sort(key=lambda cr: project(H, cr[0], cr[1])[1])  # far->near
        lines = []
        for (c, r) in occ:
            cid = FEN2ID.get(fen[r][c])
            if cid is None:
                continue
            crop = pick_crop(cid, btype)
            if crop is None:
                continue
            q = cell_quad(H, c, r, piece_height=0.20)
            composite_piece(img, crop, q, rng)
            lines.append(quad_aabb_label(q, Wimg, Himg, cid))
        # landmark labels (board features are already on the empty board)
        for name, cells in LANDMARKS.items():
            cid = NAME2ID[name]
            for (c, r) in cells:
                q = cell_quad(H, c, r, half=0.25)
                lines.append(quad_aabb_label(q, Wimg, Himg, cid))

        img = finalize(img, rng)
        stem = f"synth_{k:05d}"
        imwrite_u(str(img_dir / f"{stem}.jpg"), img)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{args.n}")

    names = ", ".join(f"'{n}'" for n in ITEM_CLASS_NAMES)
    yaml = (f"train: {sub}images\nval: {sub}images\n\n"
            f"nc: {len(ITEM_CLASS_NAMES)}\nnames: [{names}]\n")
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"\nDone: {made} images -> {out}")


if __name__ == "__main__":
    main()
