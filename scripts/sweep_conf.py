"""Sweep piece-confidence threshold on the 224-img bench (one model load).

Answers: is the 0.5 threshold dropping real pieces (MISS)? Reports exact-FEN
(mirror-tolerant) + MISS/EXTRA/WRONG totals per conf level.
"""
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boarddetection.pipeline import XiangqiRecognizer  # noqa: E402

BENCH = os.path.join(ROOT, "test", "bench")
CONFS = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]


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


def err(det, g):
    best = None
    for gg in (g, mir(g)):
        ra, rb = exp(det), exp(gg)
        m = e = w = 0
        for i in range(10):
            for j in range(9):
                a, b = ra[i][j], rb[i][j]
                if a == "." and b != ".":
                    m += 1
                elif a != "." and b == ".":
                    e += 1
                elif a != "." and b != "." and a != b:
                    w += 1
        if best is None or (m + e + w) < sum(best):
            best = (m, e, w)
    return best


def img_for(stem):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(BENCH, "images", f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None


def main():
    gt = parse_gt(os.path.join(BENCH, "ground_truth.txt"))
    rec = XiangqiRecognizer()
    items = [(s, img_for(s), g) for s, g in gt.items() if img_for(s)]
    print(f"{'conf':>5} | {'exact':>7} | {'MISS':>5} {'EXTRA':>5} {'WRONG':>5} | total")
    print("-" * 50)
    rows = []
    for conf in CONFS:
        exact = 0
        tot = Counter()
        for stem, ip, g in items:
            det = rec.recognize(ip, piece_confidence=conf).fen.split()[0]
            m, e, w = err(det, g)
            tot["M"] += m
            tot["E"] += e
            tot["W"] += w
            if m + e + w == 0:
                exact += 1
        rows.append((conf, exact, tot["M"], tot["E"], tot["W"]))
        print(f"{conf:5.2f} | {exact:3d}/{len(items)} | {tot['M']:5d} "
              f"{tot['E']:5d} {tot['W']:5d} | {tot['M']+tot['E']+tot['W']}")
    best = max(rows, key=lambda r: r[1])
    print(f"\nBEST exact: conf={best[0]} -> {best[1]}/{len(items)}")


if __name__ == "__main__":
    main()
