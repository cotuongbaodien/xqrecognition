"""Data-integrity + class-balance audit for items_v20 (+ bench leakage).

Reports:
  1. Per-class instance counts per split + imbalance (esp. confusable pairs).
  2. Cross-split leakage: same ORIGINAL board (dedup base) in >1 split.
  3. Label integrity: empty files, bad coords, invalid class ids, zero-area box.
  4. Bench<->train image leakage via average-hash (a bench board inside train
     would inflate the 243-bench score).
"""
import glob
import os
import re
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boarddetection.settings import ITEM_CLASSES  # noqa

DATA = os.path.join(ROOT, "data", "items_v20")
BENCH = os.path.join(ROOT, "test", "bench", "images")
PIECES = set(range(0, 7)) | set(range(11, 18))


def dedup_base(name):
    b = re.sub(r"\.rf\..*$", "", name)         # strip Roboflow aug hash
    b = re.sub(r"\.(jpg|jpeg|png|bmp|webp)$", "", b, flags=re.I)
    return b


def ahash(path):
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    g = cv2.resize(im, (8, 8))
    return (g > g.mean()).tobytes()


def main():
    # ---- 1. class distribution + 3. integrity ----
    print("=" * 64)
    print("1) CLASS DISTRIBUTION (instances) + 3) INTEGRITY")
    dist = {s: Counter() for s in ("train", "valid", "test")}
    empty = bad = 0
    for s in dist:
        for lp in glob.glob(os.path.join(DATA, s, "labels", "*.txt")):
            lines = [l for l in open(lp, encoding="utf-8").read().splitlines() if l.strip()]
            if not lines:
                empty += 1
                continue
            for ln in lines:
                p = ln.split()
                cid = int(p[0])
                x, y, w, h = [float(v) for v in p[1:5]]
                if cid not in ITEM_CLASSES or w <= 0 or h <= 0 or not (
                        0 <= x <= 1 and 0 <= y <= 1):
                    bad += 1
                    continue
                dist[s][cid] += 1
    tr = dist["train"]
    print(f"\n{'class':18s} {'train':>8s} {'valid':>7s} {'test':>6s}")
    for cid in sorted(ITEM_CLASSES):
        nm = ITEM_CLASSES[cid][0]
        print(f"{nm:18s} {tr[cid]:8d} {dist['valid'][cid]:7d} {dist['test'][cid]:6d}")
    print(f"\nempty label files: {empty} | bad/invalid boxes: {bad}")

    # imbalance among the 14 pieces (train)
    pc = {cid: tr[cid] for cid in PIECES if tr[cid]}
    if pc:
        mx, mn = max(pc.values()), min(pc.values())
        hi = ITEM_CLASSES[max(pc, key=pc.get)][0]
        lo = ITEM_CLASSES[min(pc, key=pc.get)][0]
        print(f"\nPIECE imbalance (train): max {hi}={mx}  min {lo}={mn}  ratio {mx/mn:.2f}x")

        def pair(a_red, a_blk, b_red, b_blk, label):
            A = tr[a_red] + tr[a_blk]
            B = tr[b_red] + tr[b_blk]
            print(f"  {label}: {A} vs {B}  -> {max(A,B)/max(1,min(A,B)):.2f}x "
                  f"({'lệch' if max(A,B)/max(1,min(A,B))>1.3 else 'cân'})")
        # confusable pairs: chariot(2,13) vs horse(5,16); general(4,15) vs elephant(3,14)
        pair(13, 2, 16, 5, "xe vs mã   ")
        pair(15, 4, 14, 3, "vua vs tượng")

    # ---- 2. cross-split leakage ----
    print("\n" + "=" * 64)
    print("2) CROSS-SPLIT LEAKAGE (cùng bàn gốc ở >1 split)")
    base_split = defaultdict(set)
    for s in ("train", "valid", "test"):
        for ip in glob.glob(os.path.join(DATA, s, "images", "*")):
            base_split[dedup_base(os.path.basename(ip))].add(s)
    leak = {b: ss for b, ss in base_split.items() if len(ss) > 1}
    print(f"bàn gốc nằm ở nhiều split: {len(leak)}")
    for b, ss in list(leak.items())[:8]:
        print(f"  {sorted(ss)}  {b[:50]}")

    # ---- 4. bench <-> train image leakage ----
    print("\n" + "=" * 64)
    print("4) BENCH <-> TRAIN image leakage (average-hash)")
    train_h = {}
    for ip in glob.glob(os.path.join(DATA, "train", "images", "*")):
        h = ahash(ip)
        if h:
            train_h.setdefault(h, os.path.basename(ip))
    hits = []
    for bp in glob.glob(os.path.join(BENCH, "*")):
        h = ahash(bp)
        if h and h in train_h:
            hits.append((os.path.basename(bp), train_h[h]))
    print(f"train images hashed: {len(train_h)} | bench checked: "
          f"{len(glob.glob(os.path.join(BENCH,'*')))}")
    print(f"BENCH ảnh trùng trong TRAIN: {len(hits)}")
    for b, t in hits[:20]:
        print(f"  bench {b}  ==  train {t[:45]}")


if __name__ == "__main__":
    main()
