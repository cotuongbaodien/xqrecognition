"""Proper GATED xe<->ma tiebreaker + net bench evaluation.

- Trains a bigger COLOR 64x64 classifier on real items_v20 xe/ma crops (cached).
- At inference (monkeypatched into item_detector.detect): for a detected xe/ma
  piece whose detector confidence is BELOW a gate (i.e. the detector is unsure),
  run the classifier; flip the piece's class only if the classifier disagrees
  with high confidence. Confident detections are never touched -> low regression.
- Measures NET exact-FEN change on the 219-board bench: fixed vs regressed.

Usage: python scripts/tiebreak_xema2.py
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "items_v20")
BENCH = os.path.join(ROOT, "test", "bench")
SZ = 64
CLS_TYPE = {5: 0, 16: 0, 2: 1, 13: 1}        # yolo id -> 0 ma / 1 xe
# (color, type) -> (class_id, fen_symbol, class_name)
REMAP = {
    ("black", 1): (2, "r", "black-chariot"), ("black", 0): (5, "n", "black-horse"),
    ("red", 1): (13, "R", "red-chariot"),    ("red", 0): (16, "N", "red-horse"),
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def prep(crop):
    g = cv2.resize(crop, (SZ, SZ)).astype(np.float32) / 255.0
    return g.transpose(2, 0, 1)        # CHW


def aug(g):
    out = [g]
    for k in (1, 2, 3):
        out.append(np.rot90(g, k, axes=(1, 2)).copy())
    out.append(g[:, :, ::-1].copy())
    out.append(g[:, ::-1, :].copy())
    return out


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(),
                               nn.Dropout(0.4), nn.Linear(128, 2))

    def forward(self, x):
        return self.c(self.f(x))


def build(split):
    cache = os.path.join(ROOT, "runs", f"xema2_{split}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["y"]
    X, y = [], []
    for ip in glob.glob(os.path.join(DATA, split, "images", "*")):
        lp = ip.replace("images", "labels", 1).rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lp):
            continue
        im = imread_u(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        for ln in open(lp, encoding="utf-8"):
            p = ln.split()
            if not p or int(p[0]) not in CLS_TYPE:
                continue
            x, yy, w, h = [float(v) for v in p[1:5]]
            x1, y1 = int((x - w / 2) * W), int((yy - h / 2) * H)
            x2, y2 = int((x + w / 2) * W), int((yy + h / 2) * H)
            cr = im[max(0, y1):y2, max(0, x1):x2]
            if cr.size == 0:
                continue
            X.append(prep(cr))
            y.append(CLS_TYPE[int(p[0])])
    X, y = np.array(X, np.float32), np.array(y)
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


# ---- FEN helpers ----
def parse_gt(p):
    g = {}
    for ln in open(p, encoding="utf-8"):
        if ":" in ln and not ln.strip().startswith("#"):
            k, v = ln.split(":", 1)
            if v.strip():
                g[k.strip()] = v.strip().split()[0]
    return g


def exp(f):
    rows = []
    for row in f.split("/"):
        c = []
        for ch in row:
            c += ["."] * int(ch) if ch.isdigit() else [ch]
        rows.append((c + ["."] * 9)[:9])
    while len(rows) < 10:
        rows.append(["."] * 9)
    return rows[:10]


def mir(f):
    return "/".join("".join(r[::-1]) for r in exp(f))


def is_exact(det, g):
    return exp(det) == exp(g) or exp(det) == exp(mir(g))


def main():
    print(f"device={DEV}  building crops (cached) ...")
    Xtr, ytr = build("train")
    Xv, yv = build("valid")
    print(f"train {Xtr.shape}  val {Xv.shape}")
    # augment train
    Xa, ya = [], []
    for g, l in zip(Xtr, ytr):
        for ga in aug(g):
            Xa.append(ga); ya.append(l)
    Xa = torch.tensor(np.array(Xa)); ya = torch.tensor(np.array(ya))
    net = Net2().to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lf = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    bs = 512
    for ep in range(14):
        net.train(); perm = torch.randperm(len(Xa)); tot = 0
        for i in range(0, len(Xa), bs):
            idx = perm[i:i + bs]
            xb = Xa[idx].to(DEV); yb = ya[idx].to(DEV)
            opt.zero_grad(); out = net(xb); loss = lf(out, yb)
            loss.backward(); opt.step(); tot += loss.item() * len(idx)
        print(f"  ep{ep + 1} loss={tot / len(Xa):.4f}")
    net.eval()
    with torch.no_grad():
        Xvt = torch.tensor(Xv).to(DEV)
        pv = net(Xvt).argmax(1).cpu().numpy()
    acc = (pv == yv).mean()
    print(f"VAL ACC (color {SZ}px): {acc:.4f}  (ma={ (pv[yv==0]==0).mean():.4f}, "
          f"xe={ (pv[yv==1]==1).mean():.4f})")

    def classify(crop):
        with torch.no_grad():
            g = torch.tensor(prep(crop)[None]).to(DEV)
            pr = torch.softmax(net(g), 1)[0].cpu().numpy()
        return int(pr.argmax()), float(pr.max())

    # ---- gated tiebreaker on bench ----
    from boarddetection.pipeline import XiangqiRecognizer
    gt = parse_gt(os.path.join(BENCH, "ground_truth.txt"))
    rec = XiangqiRecognizer()
    orig_detect = rec.item_detector.detect

    cfg = {"on": False, "GATE": 0.85, "CLF": 0.75}

    def patched(image, confidence=0.3):
        r = orig_detect(image, confidence=confidence)
        if cfg["on"]:
            H, W = image.shape[:2]
            for p in r.pieces:
                if p.class_id not in CLS_TYPE or p.confidence >= cfg["GATE"]:
                    continue
                x1, y1, x2, y2 = [int(v) for v in p.bbox]
                cr = image[max(0, y1):y2, max(0, x1):x2]
                if cr.size == 0:
                    continue
                t, cp = classify(cr)
                if t != CLS_TYPE[p.class_id] and cp >= cfg["CLF"]:
                    color = "red" if p.class_id in (13, 16) else "black"
                    cid, sym, nm = REMAP[(color, t)]
                    p.class_id, p.fen_symbol, p.class_name = cid, sym, nm
        return r
    rec.item_detector.detect = patched

    imgs = []
    for stem, g in gt.items():
        for ext in (".png", ".jpg", ".jpeg"):
            fp = os.path.join(BENCH, "images", stem + ext)
            if os.path.exists(fp):
                imgs.append((stem, fp, g)); break

    # baseline
    cfg["on"] = False
    base = {}
    for stem, fp, g in imgs:
        base[stem] = is_exact(rec.recognize(fp).fen.split()[0], g)
    b_exact = sum(base.values())
    print(f"\nBASELINE exact: {b_exact}/{len(imgs)}")

    # sweep gate settings
    for GATE, CLF in [(0.85, 0.75), (0.9, 0.8), (0.8, 0.7), (0.95, 0.85)]:
        cfg.update(on=True, GATE=GATE, CLF=CLF)
        fixed, regr = [], []
        ex = 0
        for stem, fp, g in imgs:
            ok = is_exact(rec.recognize(fp).fen.split()[0], g)
            ex += ok
            if ok and not base[stem]:
                fixed.append(stem)
            if not ok and base[stem]:
                regr.append(stem)
        print(f"GATE={GATE} CLF={CLF}: exact {ex}/{len(imgs)} "
              f"(net {ex - b_exact:+d}) | fixed={fixed} | REGRESS={regr}")


if __name__ == "__main__":
    main()
