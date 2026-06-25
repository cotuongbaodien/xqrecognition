"""Find likely MISLABELED horse<->chariot boxes in the real training data.

mã<->xe is the top WRONG on bench (71/130), bidirectional + symmetric +
class-balanced => label noise, not data scarcity. The synthetic crops are
clean (labels come from the FEN), so we train a small horse-vs-chariot
classifier on the DIGITAL crops only, then run it over every real horse/
chariot box. Where the classifier confidently disagrees with the label, the
box is flagged for review on Roboflow.

Outputs:
  test/bench/_mislabel_flagged.jpg   visual montage (src img + label -> pred)
  data/mislabel_review/flagged.txt   src image numbers + box info for Roboflow
  data/mislabel_review/crops/        the flagged crops

Usage: python scripts/find_mislabels.py [--conf 0.85]
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SZ = 48
# horse classes (5 black, 16 red) -> label 0 ; chariot (2,13) -> label 1
HORSE_NAMES = ["black-horse", "red-horse"]
CHARIOT_NAMES = ["black-chariot", "red-chariot"]
CLS_TYPE = {5: 0, 16: 0, 2: 1, 13: 1}      # yolo id -> 0 horse / 1 chariot
TYPE_NAME = {0: "ma(H)", 1: "xe(X)"}


def imread_u(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


def prep(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (SZ, SZ)).astype(np.float32) / 255.0
    return g


def aug(g, rng):
    """rotation+flip so the classifier is robust to Roboflow augmentations."""
    out = [g]
    for k in (1, 2, 3):
        out.append(np.rot90(g, k).copy())
    out.append(g[:, ::-1].copy())
    out.append(g[::-1, :].copy())
    return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16, 64),
                               nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        return self.c(self.f(x))


def load_digital():
    X, y = [], []
    rng = np.random.default_rng(0)
    for label, names in ((0, HORSE_NAMES), (1, CHARIOT_NAMES)):
        for nm in names:
            for f in glob.glob(f"{ROOT}/data/piece_crops_all/{nm}/dh*.png") + \
                     glob.glob(f"{ROOT}/data/piece_crops_all/{nm}/dl*.png"):
                im = imread_u(f, cv2.IMREAD_UNCHANGED)
                if im is None:
                    continue
                if im.ndim == 3 and im.shape[2] == 4:
                    a = im[:, :, 3:] / 255.0
                    im = (im[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
                for g in aug(prep(im), rng):
                    X.append(g)
                    y.append(label)
    return np.array(X)[:, None], np.array(y)


def train(X, y, epochs=12):
    dev = "cpu"
    torch.manual_seed(0)          # deterministic weights -> reproducible flags
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n = len(X)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, 128):
            b = idx[i:i + 128]
            opt.zero_grad()
            out = net(Xt[b])
            loss = lossf(out, yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        acc = float((net(Xt).argmax(1) == yt).float().mean())
        print(f"  ep{ep + 1} loss={tot / n:.3f} train_acc={acc:.3f}")
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.85)
    args = ap.parse_args()

    print("training horse-vs-chariot classifier on clean DIGITAL crops...")
    X, y = load_digital()
    print(f"  digital train samples (with aug): {len(X)} "
          f"(horse {int((y == 0).sum())} / chariot {int((y == 1).sum())})")
    net = train(X, y)

    outdir = f"{ROOT}/data/mislabel_review"
    os.makedirs(f"{outdir}/crops", exist_ok=True)
    flagged = []          # (srcnum, yolo_id, pred, conf, crop)
    by_src = defaultdict(list)
    imgs = glob.glob(f"{ROOT}/data/items_v19/train/images/items_v16_*")
    sm = torch.nn.Softmax(1)
    print(f"scanning {len(imgs)} real images...")
    for ip in imgs:
        # Roboflow source name = filename minus our "items_v16_" prefix and
        # Roboflow's ".rf.<hash>.<ext>" suffix (and trailing _jpg/_png tokens).
        base = os.path.basename(ip)
        base = re.sub(r"^items_v16_", "", base)
        base = re.sub(r"\.rf\..*$", "", base)
        base = re.sub(r"(_png|_jpg|_jpeg)+$", "", base)
        src = base
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
            if c not in CLS_TYPE:
                continue
            x, yy, w, h = [float(v) for v in p[1:5]]
            x1, y1 = int((x - w / 2) * W), int((yy - h / 2) * H)
            x2, y2 = int((x + w / 2) * W), int((yy + h / 2) * H)
            cr = im[max(0, y1):y2, max(0, x1):x2]
            if cr.size == 0:
                continue
            t = torch.tensor(prep(cr)[None, None], dtype=torch.float32)
            with torch.no_grad():
                prob = sm(net(t))[0].numpy()
            pred = int(prob.argmax())
            lab = CLS_TYPE[c]
            if pred != lab and prob[pred] >= args.conf:
                flagged.append((src, c, pred, float(prob[pred]), cr, lp, li))
                by_src[src].append((TYPE_NAME[lab], TYPE_NAME[pred],
                                    round(float(prob[pred]), 2)))

    print(f"\nFLAGGED {len(flagged)} suspect boxes across "
          f"{len(by_src)} source images (conf>={args.conf})")

    def montage_and_list(items, name, direction_label):
        items = sorted(items, key=lambda t: -t[3])
        # montage (top 96), tile index shown so it maps to the manifest
        show = items[:96]
        cols = 8
        rows = max(1, (len(show) + cols - 1) // cols)
        cv = np.full((rows * 96, cols * 96, 3), 255, np.uint8)
        for i, t in enumerate(show):
            cr, cf = t[4], t[3]
            tile = cv2.resize(cr, (96, 96))
            cv2.putText(tile, f"#{i}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 0, 255), 1)
            cv2.putText(tile, f"{cf:.2f}", (3, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 200), 1)
            cv[(i // cols) * 96:(i // cols) * 96 + 96,
               (i % cols) * 96:(i % cols) * 96 + 96] = tile
        cv2.imencode(".jpg", cv)[1].tofile(
            f"{ROOT}/test/bench/_mislabel_{name}.jpg")
        # manifest: tile index (montage order) -> exact source box
        import json as _json
        man = [{"tile": i, "file": os.path.relpath(t[5], ROOT).replace("\\", "/"),
                "line": t[6], "old": t[1], "conf": round(t[3], 3), "src": t[0]}
               for i, t in enumerate(show)]
        _json.dump(man, open(f"{outdir}/manifest_{name}.json", "w"), indent=1)
        per = defaultdict(int)
        for t in items:
            per[t[0]] += 1
        with open(f"{outdir}/flagged_{name}.txt", "w", encoding="utf-8") as f:
            f.write(f"# {direction_label}: {len(items)} box, "
                    f"{len(per)} anh goc (conf>={args.conf})\n")
            for src in sorted(per, key=lambda s: (-per[s], s)):
                f.write(f"anh {src}: {per[src]} box\n")
        return len(items), len(per)

    h2x = [t for t in flagged if CLS_TYPE[t[1]] == 0]   # labeled horse, pred chariot
    x2h = [t for t in flagged if CLS_TYPE[t[1]] == 1]   # labeled chariot, pred horse
    nb_h, ns_h = montage_and_list(h2x, "H2X", "label=MA nhung giong XE (TIN CAY)")
    nb_x, ns_x = montage_and_list(x2h, "X2H", "label=XE nhung giong MA (NHIEU NHIEU - 車 co)")

    # flip manifest for H2X only (horse->chariot): 5->2 (black), 16->13 (red)
    FLIP = {5: 2, 16: 13}
    h2x_sorted = sorted(h2x, key=lambda t: -t[3])
    flips = [{"file": os.path.relpath(t[5], ROOT).replace("\\", "/"),
              "line": t[6], "old": t[1], "new": FLIP[t[1]],
              "conf": round(t[3], 3), "src": t[0]} for t in h2x_sorted]
    import json
    with open(f"{outdir}/to_flip.json", "w", encoding="utf-8") as f:
        json.dump(flips, f, indent=1)
    for thr in (0.92, 0.95, 0.97, 0.99, 0.995):
        nh = sum(1 for t in h2x if t[3] >= thr)
        nx = sum(1 for t in x2h if t[3] >= thr)
        print(f"  conf>={thr}: H2X={nh}  X2H={nx}")
    print(f"\n[TIN CAY] H2X (label ma, la xe): {nb_h} box / {ns_h} anh -> _mislabel_H2X.jpg")
    print(f"[NHIEU]   X2H (label xe, la ma): {nb_x} box / {ns_x} anh -> _mislabel_X2H.jpg")
    print(f"-> {outdir}/to_flip.json ({len(flips)} box horse->chariot, deterministic)")


if __name__ == "__main__":
    main()
