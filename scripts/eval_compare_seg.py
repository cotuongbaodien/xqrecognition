"""Compare board-seg models on the 86-image test set (mirror-tolerant FEN).

Runs the full pipeline twice on the same images — once with the deployed
board_seg.pt, once with a candidate model — and reports exact-match and
mirror-tolerant exact-match counts plus a per-image regression/improvement
list. The items model is loaded once and shared; only the board segmenter is
swapped.

Usage:
    python scripts/eval_compare_seg.py --candidate models/backups/board_seg_v5.pt
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from boarddetection.pipeline import XiangqiRecognizer
from boarddetection.board_segmenter import BoardSegmenter
from boarddetection.settings import MODELS_DIR

TEST_DIR = ROOT / "test"
GT_PATH = TEST_DIR / "ground_truth.txt"


def parse_gt(path):
    gt = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, fen = line.split(":", 1)
        gt[key.strip()] = fen.strip().split()[0]
    return gt


def expand_rows(fen):
    rows = []
    for row in fen.split("/"):
        cells = []
        for ch in row:
            if ch.isdigit():
                cells += ["."] * int(ch)
            else:
                cells.append(ch)
        rows.append((cells + ["."] * 9)[:9])
    while len(rows) < 10:
        rows.append(["."] * 9)
    return rows[:10]


def mirror_fen(fen):
    return "/".join("".join(r[::-1]) for r in expand_rows(fen))


def board_eq(a, b):
    return expand_rows(a) == expand_rows(b)


def match(det, gt):
    """Return 'exact', 'mirror', or 'wrong'."""
    if board_eq(det, gt):
        return "exact"
    if board_eq(det, mirror_fen(gt)):
        return "mirror"
    return "wrong"


def run_model(recognizer, seg_path, images):
    if seg_path is None:
        recognizer.board_segmenter = None
    else:
        recognizer.board_segmenter = BoardSegmenter(str(seg_path))
    out = {}
    for img in images:
        try:
            res = recognizer.recognize(str(img))
            out[img.stem] = res.fen.split()[0]
        except Exception as e:
            out[img.stem] = f"ERR:{e}"
    return out


def summarize(name, det, gt):
    exact = mirror = wrong = 0
    for k, g in gt.items():
        if k not in det:
            wrong += 1
            continue
        m = match(det[k], g)
        exact += m == "exact"
        mirror += m == "mirror"
        wrong += m == "wrong"
    print(f"{name}: exact={exact}  +mirror={mirror}  (mirror-tolerant={exact + mirror})  wrong={wrong}  /{len(gt)}")
    return {k: match(det.get(k, ""), g) for k, g in gt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=str(MODELS_DIR.parent.parent / "models" / "backups" / "board_seg_v5.pt"))
    ap.add_argument("--baseline", default=str(MODELS_DIR / "board_seg.pt"))
    args = ap.parse_args()

    gt = parse_gt(GT_PATH)
    images = sorted(
        [f for f in TEST_DIR.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 1e9,
    )
    print(f"{len(images)} images, {len(gt)} GT entries\n")

    recognizer = XiangqiRecognizer()

    print("Running BASELINE...")
    base = run_model(recognizer, args.baseline, images)
    print("Running CANDIDATE...")
    cand = run_model(recognizer, args.candidate, images)

    print("\n=== RESULTS (mirror-tolerant) ===")
    bstat = summarize("baseline ", base, gt)
    cstat = summarize("candidate", cand, gt)

    print("\n=== CHANGES (candidate vs baseline) ===")
    order = {"exact": 0, "mirror": 1, "wrong": 2}
    for k in sorted(gt, key=lambda k: int(k) if k.isdigit() else 1e9):
        b, c = bstat[k], cstat[k]
        if b != c:
            arrow = "IMPROVE" if order[c] < order[b] else "REGRESS"
            print(f"  {k:4s} {b:6s} -> {c:6s}  {arrow}")


if __name__ == "__main__":
    main()
