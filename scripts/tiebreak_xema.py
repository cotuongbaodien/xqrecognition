"""Experiment: a xe<->ma tiebreaker classifier trained on REAL crops.

The detector still confuses chariot(xe)<->horse(ma) on a few board styles even
with clean labels. This trains the small Net (from find_mislabels) on the REAL
items_v20 crops (correct labels), then checks (a) held-out accuracy and (b)
whether it corrects the detector on the bench's xe<->ma fails (194-199).

Idea at inference: when the detector says xe or ma, run this classifier; if it
disagrees confidently, flip. Here we just measure if that would help.

Usage: python scripts/tiebreak_xema.py
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from find_mislabels import Net, prep, aug, SZ, CLS_TYPE  # noqa: E402

DATA = os.path.join(ROOT, "data", "items_v20")
BENCH = os.path.join(ROOT, "test", "bench")
FAIL = ["194", "195", "197", "198", "199"]


def imread_u(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def build_real(split):
    """Crop every xe/ma box from a split -> (crops, labels, src) grouped by img."""
    items = []   # (gray_prepped, label, base)
    for ip in glob.glob(os.path.join(DATA, split, "images", "*")):
        lp = ip.replace("images", "labels", 1).rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lp):
            continue
        im = imread_u(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        base = os.path.basename(ip)
        for ln in open(lp, encoding="utf-8"):
            p = ln.split()
            if not p:
                continue
            cid = int(p[0])
            if cid not in CLS_TYPE:
                continue
            x, y, w, h = [float(v) for v in p[1:5]]
            x1, y1 = int((x - w / 2) * W), int((y - h / 2) * H)
            x2, y2 = int((x + w / 2) * W), int((y + h / 2) * H)
            cr = im[max(0, y1):y2, max(0, x1):x2]
            if cr.size == 0:
                continue
            items.append((prep(cr), CLS_TYPE[cid], base))
    return items


def main():
    print("Building real xe/ma crops from items_v20 ...")
    train_items = build_real("train")
    val_items = build_real("valid")
    print(f"train boxes: {len(train_items)} | val boxes: {len(val_items)}")

    # train set: augment (rot/flip) like the digital path
    Xtr, ytr = [], []
    for g, lab, _ in train_items:
        for ga in aug(g, None):
            Xtr.append(ga)
            ytr.append(lab)
    Xtr = torch.tensor(np.array(Xtr)[:, None], dtype=torch.float32)
    ytr = torch.tensor(ytr)
    print(f"train tensors: {tuple(Xtr.shape)}  (horse={int((ytr==0).sum())}, "
          f"chariot={int((ytr==1).sum())})")

    net = Net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    bs = 256
    for ep in range(12):
        net.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"  ep{ep + 1} loss={tot / len(Xtr):.4f}")

    # held-out val accuracy (no aug)
    net.eval()
    Xv = torch.tensor(np.array([g for g, _, _ in val_items])[:, None],
                      dtype=torch.float32)
    yv = torch.tensor([lab for _, lab, _ in val_items])
    with torch.no_grad():
        sm = torch.softmax(net(Xv), 1)
        pred = sm.argmax(1)
    acc = (pred == yv).float().mean().item()
    # per-class
    for lab, nm in ((0, "ma/horse"), (1, "xe/chariot")):
        m = yv == lab
        a = (pred[m] == yv[m]).float().mean().item() if m.any() else 0
        print(f"  val acc {nm:12s}: {a:.4f}  (n={int(m.sum())})")
    print(f"VAL ACC (xe vs ma): {acc:.4f}  on {len(val_items)} real crops")

    # ---- test on the bench xe<->ma fails (194-199) ----
    from boarddetection.pipeline import XiangqiRecognizer
    rec = XiangqiRecognizer()
    print("\n=== tiebreaker on bench fails 194-199 ===")
    flips = agree = 0
    for stem in FAIL:
        ip = None
        for ext in (".png", ".jpg", ".jpeg"):
            if os.path.exists(os.path.join(BENCH, "images", stem + ext)):
                ip = os.path.join(BENCH, "images", stem + ext)
        if ip is None:
            continue
        im = imread_u(ip)
        res = rec.recognize(ip)
        for p in res.pieces:
            if p.class_id not in CLS_TYPE:
                continue
            x1, y1, x2, y2 = [int(v) for v in p.bbox]
            cr = im[max(0, y1):y2, max(0, x1):x2]
            if cr.size == 0:
                continue
            with torch.no_grad():
                g = torch.tensor(prep(cr)[None, None], dtype=torch.float32)
                prob = torch.softmax(net(g), 1)[0]
            clf = int(prob.argmax())
            det = CLS_TYPE[p.class_id]
            if clf != det and float(prob[clf]) > 0.6:
                flips += 1
                print(f"  {stem}: detector={('xe' if det else 'ma')}  "
                      f"-> CLF says {('xe' if clf else 'ma')} "
                      f"(p={float(prob[clf]):.2f}, det_conf={p.confidence:.2f})")
            else:
                agree += 1
    print(f"\nagree={agree}  flips={flips} (CLF muon lat khi detector sai xe<->ma)")


if __name__ == "__main__":
    main()
