"""Analyze the bench boards the deployed items model still gets wrong.

Runs the full pipeline on test/bench, finds non-exact boards (mirror-tolerant),
and for each dumps the per-cell errors (MISS / EXTRA / WRONG + position),
classifies a likely root cause (grid/orientation vs piece misclass), and builds
a labelled montage of the failing board images.

Out: runs/v20_fail_analysis.txt  +  runs/v20_fail_montage.jpg
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boarddetection.pipeline import XiangqiRecognizer  # noqa: E402

BENCH = os.path.join(ROOT, "test", "bench")
# piece letter -> readable
NAME = {"r": "xe", "n": "ma", "c": "phao", "a": "si", "b": "tuong(voi)",
        "k": "tuong(tg)", "p": "tot",
        "R": "XE", "N": "MA", "C": "PHAO", "A": "SI", "B": "TUONG(voi)",
        "K": "TUONG(tg)", "P": "TOT"}


def parse_gt(p):
    g = {}
    for ln in open(p, encoding="utf-8"):
        if ":" in ln and not ln.strip().startswith("#"):
            k, v = ln.split(":", 1)
            if v.strip():
                g[k.strip()] = v.strip()
    return g


def exp(f):
    rows = []
    for row in f.split()[0].split("/"):
        c = []
        for ch in row:
            c += ["."] * int(ch) if ch.isdigit() else [ch]
        rows.append((c + ["."] * 9)[:9])
    while len(rows) < 10:
        rows.append(["."] * 9)
    return rows[:10]


def mir(f):
    return "/".join("".join(r[::-1]) for r in exp(f))


def diff(det, g):
    """Return (errs list, counts) for det vs g already expanded."""
    ra, rb = exp(det), exp(g)
    errs = []
    for i in range(10):
        for j in range(9):
            a, b = ra[i][j], rb[i][j]
            if a == b:
                continue
            if a == "." and b != ".":
                errs.append(("MISS", i, j, None, b))
            elif a != "." and b == ".":
                errs.append(("EXTRA", i, j, a, None))
            else:
                errs.append(("WRONG", i, j, a, b))
    return errs


def img_for(stem):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(BENCH, "images", f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None


def main():
    gt = parse_gt(os.path.join(BENCH, "ground_truth.txt"))
    rec = XiangqiRecognizer()
    fails = []
    for stem, g in gt.items():
        ip = img_for(stem)
        if ip is None:
            continue
        det = rec.recognize(ip).fen.split()[0]
        # mirror-tolerant: pick orientation with fewest errors
        best = None
        for gg in (g, mir(g)):
            e = diff(det, gg)
            if best is None or len(e) < len(best[1]):
                best = (gg, e)
        gg, errs = best
        if errs:
            fails.append((stem, ip, det, gg, errs))

    # ---- text report ----
    out = open(os.path.join(ROOT, "runs", "v20_fail_analysis.txt"), "w",
               encoding="utf-8")
    tot = Counter()
    wrong_pairs = Counter()
    cat = Counter()
    for stem, ip, det, gg, errs in fails:
        c = Counter(e[0] for e in errs)
        tot.update(c)
        # classify: heavy MISS+EXTRA & few WRONG often => grid/orientation shift
        nmiss, nextra, nwrong = c["MISS"], c["EXTRA"], c["WRONG"]
        if nmiss + nextra >= 6 and nwrong <= 2:
            category = "GRID/ORIENTATION? (nhieu miss+extra)"
        elif nwrong >= max(1, nmiss + nextra):
            category = "PIECE MISCLASS"
        else:
            category = "MIXED"
        cat[category] += 1
        out.write(f"\n=== {stem}  [{category}]  "
                  f"MISS={nmiss} EXTRA={nextra} WRONG={nwrong} ===\n")
        out.write(f"  det: {det}\n  gt : {gg}\n")
        for t, i, j, a, b in errs:
            if t == "WRONG":
                wrong_pairs[f"{NAME.get(a, a)}->{NAME.get(b, b)}"] += 1
                out.write(f"  WRONG @({i},{j}): nhan {NAME.get(a, a)} "
                          f"nhung GT la {NAME.get(b, b)}\n")
            elif t == "MISS":
                out.write(f"  MISS  @({i},{j}): sot quan {NAME.get(b, b)}\n")
            else:
                out.write(f"  EXTRA @({i},{j}): du quan {NAME.get(a, a)}\n")

    out.write("\n\n========== TONG KET ==========\n")
    out.write(f"So ban sai: {len(fails)}/{len(gt)}\n")
    out.write(f"Phan loai: {dict(cat)}\n")
    out.write(f"Tong loi: {dict(tot)}\n")
    out.write("Top cap WRONG (nhan->GT):\n")
    for k, v in wrong_pairs.most_common(15):
        out.write(f"  {v:3d}  {k}\n")
    out.close()

    # ---- montage ----
    n = len(fails)
    cols = 6
    rows = (n + cols - 1) // cols
    TH = 240
    canvas = np.full((rows * TH, cols * TH, 3), 30, np.uint8)
    for k, (stem, ip, det, gg, errs) in enumerate(fails):
        im = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        s = (TH - 30) / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        ih, iw = im.shape[:2]
        r, cc = k // cols, k % cols
        y0, x0 = r * TH, cc * TH
        canvas[y0:y0 + ih, x0:x0 + iw] = im
        c = Counter(e[0] for e in errs)
        lab = f"{stem} M{c['MISS']}E{c['EXTRA']}W{c['WRONG']}"
        cv2.putText(canvas, lab, (x0 + 2, y0 + TH - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.imencode(".jpg", canvas)[1].tofile(
        os.path.join(ROOT, "runs", "v20_fail_montage.jpg"))

    # ---- console summary ----
    print(f"FAILS: {len(fails)}/{len(gt)}")
    print(f"Categories: {dict(cat)}")
    print(f"Total cell errors: {dict(tot)}")
    print("Top WRONG pairs (det->GT):")
    for k, v in wrong_pairs.most_common(12):
        print(f"  {v:3d}  {k}")
    print("\n-> runs/v20_fail_analysis.txt + runs/v20_fail_montage.jpg")


if __name__ == "__main__":
    main()
