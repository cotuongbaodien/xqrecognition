"""Per-class label-review galleries for manual auditing.

For each target class, crop every labeled box (deduped to one copy per source
image), sort SUSPICION-FIRST (a clean horse-vs-chariot classifier scores how
likely the crop is the OTHER class) so mislabels cluster on the first sheets,
and lay them out on numbered montage sheets. A manifest maps (class, index)
back to the exact (label file, line) so the user's corrections can be applied.

Folders: data/label_review/<tag>/sheet_NNN.jpg  +  manifest_<tag>.json
Tags: xeden/xedo (chariot), maden/mado (horse), phaoden/phaodo (cannon).

Usage: python scripts/review_gallery.py
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_mislabels import Net, load_digital, train, prep, imread_u  # noqa: E402

# class id -> (folder tag, suspicion mode). mode: 'chariot'/'horse' use the
# classifier (P of other type); 'none' = natural order.
# tag legend: xe=chariot ma=horse phao=cannon si=advisor tot=soldier
#   tuong=ELEPHANT(tượng/象)  soai=GENERAL(tướng/將帥)   -den=black -do=red
TARGETS = {
    2:  ("xeden", "chariot"),  13: ("xedo", "chariot"),
    5:  ("maden", "horse"),    16: ("mado", "horse"),
    1:  ("phaoden", "none"),   12: ("phaodo", "none"),
    0:  ("siden", "none"),     11: ("sido", "none"),
    3:  ("tuongden", "none"),  14: ("tuongdo", "none"),
    4:  ("soaiden", "none"),   15: ("soaido", "none"),
    6:  ("totden", "none"),    17: ("totdo", "none"),
}
SZ = 80
COLS = 10
ROWS = 10
PER = COLS * ROWS
# priority order -> numeric folder prefix (1.xeden, 2.maden, ...)
ORDER = ["xeden", "maden", "xedo", "mado", "phaoden", "phaodo", "tuongden",
         "tuongdo", "totden", "totdo", "siden", "sido", "soaiden", "soaido"]
PREFIX = {tag: i + 1 for i, tag in enumerate(ORDER)}


def main():
    only = set(sys.argv[1:])   # optional: only regen these tags
    print("training horse-vs-chariot classifier (for suspicion sort)...")
    X, y = load_digital()
    net = train(X, y)
    sm = torch.nn.Softmax(1)

    # dedupe real sources (one augmentation per original)
    seen = {}
    for ip in glob.glob(f"{ROOT}/data/items_v19/*/images/items_v16_*"):
        b = os.path.basename(ip)
        base = re.sub(r"^items_v16_", "", b)
        base = re.sub(r"\.rf\..*$", "", base)
        seen.setdefault(base, ip)
    print(f"deduped sources: {len(seen)}")

    # collect crops per target class
    per_class = defaultdict(list)   # cid -> [(suspicion, crop, file, line, src)]
    for base, ip in seen.items():
        lp = ip.replace("images", "labels", 1).rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lp):
            continue
        im = imread_u(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        for li, ln in enumerate(open(lp)):
            p = ln.split()
            if not p:
                continue
            c = int(p[0])
            if c not in TARGETS:
                continue
            if only and TARGETS[c][0] not in only:
                continue
            x, yy, w, h = [float(v) for v in p[1:5]]
            x1, y1 = int((x - w / 2) * W), int((yy - h / 2) * H)
            x2, y2 = int((x + w / 2) * W), int((yy + h / 2) * H)
            cr = im[max(0, y1):y2, max(0, x1):x2]
            if cr.size == 0:
                continue
            mode = TARGETS[c][1]
            susp = 0.0
            if mode in ("chariot", "horse"):
                t = torch.tensor(prep(cr)[None, None], dtype=torch.float32)
                with torch.no_grad():
                    prob = sm(net(t))[0].numpy()
                # suspicion = prob of the OTHER type
                susp = float(prob[1] if mode == "horse" else prob[0])
            per_class[c].append((susp, cv2.resize(cr, (SZ, SZ)),
                                 os.path.relpath(lp, ROOT).replace("\\", "/"),
                                 li, base))

    outroot = f"{ROOT}/data/label_review"
    for c, (tag, mode) in TARGETS.items():
        if only and tag not in only:
            continue
        items = per_class.get(c, [])
        items.sort(key=lambda t: -t[0])    # suspicion-first
        d = f"{outroot}/{PREFIX.get(tag, 99)}.{tag}"
        os.makedirs(d, exist_ok=True)
        for old in glob.glob(f"{d}/sheet_*.jpg"):
            os.remove(old)
        manifest = []
        nsheets = (len(items) + PER - 1) // PER
        for s in range(nsheets):
            chunk = items[s * PER:(s + 1) * PER]
            cv = np.full((ROWS * SZ, COLS * SZ, 3), 255, np.uint8)
            for k, (susp, cr, f, li, src) in enumerate(chunk):
                gidx = s * PER + k
                tile = cr.copy()
                cv2.putText(tile, str(gidx), (2, 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                cv[(k // COLS) * SZ:(k // COLS) * SZ + SZ,
                   (k % COLS) * SZ:(k % COLS) * SZ + SZ] = tile
                manifest.append({"idx": gidx, "file": f, "line": li,
                                 "susp": round(susp, 3), "src": src})
            cv2.imencode(".jpg", cv)[1].tofile(f"{d}/sheet_{s + 1:03d}.jpg")
        json.dump(manifest, open(f"{outroot}/manifest_{tag}.json", "w"), indent=1)
        flag = sum(1 for it in items if it[0] >= 0.85) if mode != "none" else 0
        print(f"  {tag:8s} ({_name(c)}): {len(items)} crop, {nsheets} sheet"
              + (f", ~{flag} nghi (susp>=0.85) o sheet dau" if flag else ""))
    print(f"\n-> {outroot}/<tag>/sheet_NNN.jpg + manifest_<tag>.json")
    print("Review: moi tile co so #idx. Bao loi kieu: 'xeden 5,12 la ma den'.")


def _name(c):
    import sys as _s
    _s.path.insert(0, ROOT)
    from boarddetection.settings import ITEM_CLASSES
    return ITEM_CLASSES[c][0]


if __name__ == "__main__":
    main()
