"""Build board_seg_v6_nonempty: drop the empty (0-piece) boards from train+valid.

Runs items.pt to count pieces per image; boards with 0 pieces (the empty_boards
collection, ~30% of v6) are out-of-distribution vs prod (always piece-full).
Copies only piece-bearing boards. Also writes the empty list for the synth step.
"""
import glob
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ultralytics import YOLO                                   # noqa
from boarddetection.settings import PIECE_CLASS_IDS, ITEMS_MODEL  # noqa

SRC = os.path.join(ROOT, "data", "board_seg_v6")
DST = os.path.join(ROOT, "data", "board_seg_v6_nonempty")


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def main():
    m = YOLO(str(ITEMS_MODEL))
    empties = []
    for split in ("train", "valid"):
        si = os.path.join(SRC, split, "images")
        sl = os.path.join(SRC, split, "labels")
        di = os.path.join(DST, split, "images")
        dl = os.path.join(DST, split, "labels")
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        kept = dropped = 0
        for ip in glob.glob(os.path.join(si, "*")):
            im = imread_u(ip)
            if im is None:
                continue
            r = m(im, conf=0.25, imgsz=960, verbose=False)[0]
            npc = sum(1 for i in range(len(r.boxes))
                      if int(r.boxes.cls[i]) in PIECE_CLASS_IDS) \
                if r.boxes is not None else 0
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(sl, stem + ".txt")
            if npc == 0:
                empties.append((split, os.path.basename(ip)))
                dropped += 1
                continue
            shutil.copy(ip, os.path.join(di, os.path.basename(ip)))
            if os.path.exists(lp):
                shutil.copy(lp, os.path.join(dl, stem + ".txt"))
            kept += 1
        print(f"{split}: kept {kept}, dropped {dropped} empty")
    # copy test split unchanged
    for sub in ("images", "labels"):
        s = os.path.join(SRC, "test", sub)
        d = os.path.join(DST, "test", sub)
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(s, "*")):
            shutil.copy(f, os.path.join(d, os.path.basename(f)))
    # data.yaml
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {DST.replace(os.sep, '/')}\n"
                "train: train/images\nval: valid/images\ntest: test/images\n"
                "nc: 2\nnames: ['xiangqi-board', 'xiangqi-palace']\n")
    # save empty list for the synth step
    with open(os.path.join(ROOT, "runs", "empty_boards_v6.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(f"{s}/{n}" for s, n in empties) + "\n")
    print(f"\n{len(empties)} empty boards -> runs/empty_boards_v6.txt")
    print(f"non-empty dataset: {DST}")


if __name__ == "__main__":
    main()
