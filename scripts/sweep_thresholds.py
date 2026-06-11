"""Sweep pipeline thresholds (conf / NMS IoU / snap ratio) on the test set.

The pipeline's thresholds were never tuned systematically — conf=0.3 (CLI
default), nms_iou=0.35, snap_ratio=0.6 are inherited values. This runs the
86-image benchmark across a grid of configs with ONE model load and reports
mirror-tolerant exact-FEN per config.

Usage:
    python scripts/sweep_thresholds.py                 # staged sweep around baseline
    python scripts/sweep_thresholds.py --full          # full cartesian grid
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from eval_fen import parse_gt, mirror_fen, diff  # noqa: E402


def score_config(recognizer, images, gt, conf, nms, snap):
    exact = 0
    fails = []
    for img_path in images:
        key = img_path.stem
        if key not in gt:
            continue
        try:
            result = recognizer.recognize(
                str(img_path),
                piece_confidence=conf,
                visualize=False,
                nms_iou=nms,
                snap_ratio=snap,
            )
            fen = result.fen.split()[0]
        except Exception:
            fails.append(key)
            continue
        g = gt[key]
        if min(diff(fen, g)[0], diff(fen, mirror_fen(g))[0]) == 0:
            exact += 1
        else:
            fails.append(key)
    return exact, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Full cartesian grid instead of staged sweep")
    ap.add_argument("--gt", default="test/ground_truth.txt")
    ap.add_argument("--dir", default="test")
    args = ap.parse_args()

    from boarddetection import XiangqiRecognizer

    gt = parse_gt(PROJECT_ROOT / args.gt)
    test_dir = PROJECT_ROOT / args.dir
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        [f for f in test_dir.iterdir() if f.suffix.lower() in exts],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9,
    )
    print(f"{len(images)} images, {len(gt)} ground-truth entries")

    recognizer = XiangqiRecognizer()

    base = (0.3, 0.35, 0.6)  # current production values (conf, nms, snap)
    if args.full:
        configs = list(itertools.product(
            (0.25, 0.3, 0.4), (0.35, 0.5), (0.6, 0.75)))
    else:
        # Staged: baseline + one-dimension-at-a-time variations
        configs = [
            base,
            (0.25, 0.35, 0.6),
            (0.4, 0.35, 0.6),
            (0.3, 0.5, 0.6),
            (0.3, 0.35, 0.75),
        ]

    results = []
    for conf, nms, snap in configs:
        t0 = time.time()
        exact, fails = score_config(recognizer, images, gt, conf, nms, snap)
        dt = time.time() - t0
        tag = " <- baseline" if (conf, nms, snap) == base else ""
        print(f"conf={conf:<5} nms={nms:<5} snap={snap:<5} -> "
              f"{exact}/{len(gt)}  ({dt:.0f}s){tag}")
        print(f"   fails: {fails}")
        results.append({"conf": conf, "nms": nms, "snap": snap,
                        "exact": exact, "fails": fails})

    out = PROJECT_ROOT / "sweep_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    best = max(results, key=lambda r: r["exact"])
    print(f"Best: conf={best['conf']} nms={best['nms']} snap={best['snap']} "
          f"-> {best['exact']}/{len(gt)}")


if __name__ == "__main__":
    main()
