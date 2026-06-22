"""Remove flagged (wrong-grid) boards from the synth-generation source.

Deletes the board image, its QC overlay, and its corners.json entry from
data/empty_boards/. Leaves data/board_seg_pool/ untouched (hard boards are
still useful for training board-seg later).

Usage: python scripts/rm_board.py 16 20 21
"""
import glob
import json
import os
import sys

BASE = "data/empty_boards"


def main():
    nums = [int(x) for x in sys.argv[1:]]
    if not nums:
        print("usage: python scripts/rm_board.py <num> [num ...]")
        return
    cj = f"{BASE}/corners.json"
    d = json.load(open(cj, encoding="utf-8"))
    for num in nums:
        s = f"{num:03d}"
        for p in glob.glob(f"{BASE}/{s}.*"):
            if os.path.isfile(p):
                os.remove(p)
        for p in glob.glob(f"{BASE}/_grid_qc/{s}.*"):
            os.remove(p)
        for k in list(d):
            if k.split("/")[-1].startswith(s + "."):
                del d[k]
    json.dump(d, open(cj, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    imgs = [f for f in os.listdir(BASE)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"removed {nums} | empty_boards remaining: {len(imgs)}")


if __name__ == "__main__":
    main()
