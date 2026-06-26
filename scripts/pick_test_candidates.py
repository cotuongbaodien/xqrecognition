"""Pick test-set candidates from the week's staged boards.

Surfaces a diverse, piece-FULL sample (stratified across weeks, fuller boards
first) for the human to FEN-label and promote into test/bench. Renders a numbered
montage so you can eyeball which to keep. Promotion (set_gt + move to bench +
REMOVE from staging so it never leaks into train) is done after you give FENs.

  python scripts/pick_test_candidates.py --n 20
  -> weekly/test_candidates/{_montage.jpg, cand_NN.<ext>, manifest.json}
"""
import argparse
import glob
import json
import os
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(ROOT, "weekly", "staging")
OUT = os.path.join(ROOT, "weekly", "test_candidates")
PIECE_IDS = set(range(0, 7)) | set(range(11, 18))   # 14 pieces (skip 7-10 landmarks)


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def week_of(name):
    # staged name: weekly_<week>_<stem>
    parts = name.split("_", 2)
    return parts[1] if len(parts) >= 3 else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-pieces", type=int, default=14,
                    help="skip boards with fewer pieces (partial/bad)")
    args = ap.parse_args()

    rows = []
    for ip in glob.glob(os.path.join(STAGE, "images", "*")):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(STAGE, "labels", stem + ".txt")
        if not os.path.exists(lp):
            continue
        npc = sum(1 for ln in open(lp, encoding="utf-8")
                  if ln.split() and int(ln.split()[0]) in PIECE_IDS)
        if npc < args.min_pieces:
            continue
        rows.append((ip, week_of(os.path.basename(ip)), npc))
    if not rows:
        sys.exit("No staged boards (run weekly_ingest.py first).")

    # group by week, fuller boards first, round-robin across weeks for diversity
    by_week = defaultdict(list)
    for ip, w, npc in sorted(rows, key=lambda r: -r[2]):
        by_week[w].append(ip)
    weeks = sorted(by_week)
    picks, i = [], 0
    while len(picks) < args.n and any(by_week.values()):
        w = weeks[i % len(weeks)]
        if by_week[w]:
            picks.append(by_week[w].pop(0))
        i += 1

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    # copy + build numbered montage
    TH = 260
    cols = 5
    rowsN = (len(picks) + cols - 1) // cols
    canvas = np.full((rowsN * TH, cols * TH, 3), 40, np.uint8)
    manifest = []
    for idx, ip in enumerate(picks):
        ext = os.path.splitext(ip)[1].lower()
        dst = f"cand_{idx:02d}{ext}"
        shutil.copy(ip, os.path.join(OUT, dst))
        manifest.append({"idx": idx, "cand": dst,
                         "staged": os.path.basename(ip)})
        im = imread_u(ip)
        if im is None:
            continue
        h, w = im.shape[:2]
        s = min(TH / h, TH / w)
        th = cv2.resize(im, (int(w * s), int(h * s)))
        yo, xo = (idx // cols) * TH, (idx % cols) * TH
        canvas[yo:yo + th.shape[0], xo:xo + th.shape[1]] = th
        cv2.putText(canvas, str(idx), (xo + 4, yo + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (xo + 4, yo + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 220, 255), 2, cv2.LINE_AA)
    cv2.imencode(".jpg", canvas)[1].tofile(os.path.join(OUT, "_montage.jpg"))
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(f"{len(picks)} candidates -> weekly/test_candidates/_montage.jpg")
    print("Xem montage, chon idx muon lam test, bao FEN tung ban -> toi promote "
          "vao test/bench (va go khoi staging de khong leak train).")


if __name__ == "__main__":
    main()
