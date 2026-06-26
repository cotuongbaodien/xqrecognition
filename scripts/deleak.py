"""Remove bench-leaked boards from items_v20/train so test/bench is held-out.

For every bench board, find near-identical train images (dHash-256, Hamming
<= THRESH), then QUARANTINE (move, not delete -> reversible) ALL train images
sharing that original's dedup-base (catches Roboflow augmentation siblings too).
After this the 243-board bench is disjoint from train -> the next retrain gives
an honest held-out number.

  python scripts/deleak.py            # dry-run: report only
  python scripts/deleak.py --apply    # move leaked train imgs to quarantine
"""
import argparse
import glob
import os
import re
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data", "items_v20", "train")
BENCH = os.path.join(ROOT, "test", "bench", "images")
QUAR = os.path.join(ROOT, "data", "items_v20", "_quarantine_bench_leak")
THRESH = 6
_LUT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def dhash(p, s=16):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    g = cv2.resize(im, (s + 1, s)).astype(np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def base(name):
    b = re.sub(r"\.rf\..*$", "", name)
    return re.sub(r"\.(jpg|jpeg|png|bmp|webp)$", "", b, flags=re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--thresh", type=int, default=THRESH)
    args = ap.parse_args()

    tr_paths, tr_bases, tr_h = [], [], []
    for ip in glob.glob(os.path.join(TRAIN, "images", "*")):
        h = dhash(ip)
        if h is not None:
            tr_paths.append(ip)
            tr_bases.append(base(os.path.basename(ip)))
            tr_h.append(h)
    H = np.array(tr_h)                              # [N,32]
    tr_bases = np.array(tr_bases)
    print(f"train hashed: {len(tr_paths)}")

    leaked_bases, matched = set(), 0
    for bp in glob.glob(os.path.join(BENCH, "*")):
        hb = dhash(bp)
        if hb is None:
            continue
        ham = _LUT[np.bitwise_xor(H, hb)].sum(1)
        hit = ham <= args.thresh
        if hit.any():
            matched += 1
            leaked_bases.update(np.unique(tr_bases[hit]).tolist())

    # all train images sharing a leaked base (incl augmentation siblings)
    to_move = [tr_paths[i] for i in range(len(tr_paths))
               if tr_bases[i] in leaked_bases]
    print(f"\nbench boards matched in train (ham<={args.thresh}): {matched}/243")
    print(f"leaked original boards (dedup bases): {len(leaked_bases)}")
    print(f"train images to quarantine (incl augs): {len(to_move)} "
          f"of {len(tr_paths)} ({100*len(to_move)/len(tr_paths):.1f}%)")

    if not args.apply:
        print("\n(dry-run) -> rerun with --apply to move them to "
              "data/items_v20/_quarantine_bench_leak/")
        return
    os.makedirs(os.path.join(QUAR, "images"), exist_ok=True)
    os.makedirs(os.path.join(QUAR, "labels"), exist_ok=True)
    n = 0
    for ip in to_move:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(TRAIN, "labels", stem + ".txt")
        shutil.move(ip, os.path.join(QUAR, "images", os.path.basename(ip)))
        if os.path.exists(lp):
            shutil.move(lp, os.path.join(QUAR, "labels", stem + ".txt"))
        n += 1
    print(f"\nQUARANTINED {n} train images -> {QUAR}")
    print(f"train now: {len(tr_paths) - n} images. Bench 243 is now held-out.")
    print("Reversible: move files back from _quarantine_bench_leak/ if needed.")


if __name__ == "__main__":
    main()
