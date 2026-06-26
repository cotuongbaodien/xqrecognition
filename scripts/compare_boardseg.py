"""Compare board_seg models on the bench, split STRAIGHT (1-219) vs SKEWED (226-249).

Swaps each candidate into boarddetection/models/board_seg.pt, runs the full v20
pipeline on every board, and reports exact-FEN for the straight set and the
skewed set separately. Restores the original board_seg.pt at the end.

Usage: python scripts/compare_boardseg.py
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BENCH = os.path.join(ROOT, "test", "bench")
PROD = os.path.join(ROOT, "boarddetection", "models", "board_seg.pt")
SKEWED = {f"{n:03d}" for n in range(226, 250)}
MODELS = {
    "OLD (prod, v5)":        os.path.join(ROOT, "models", "backups", "board_seg_predeploy_2026-06-26.pt"),
    "v6 NONEMPTY (drop 75)": os.path.join(ROOT, "models", "backups", "board_seg_v6_nonempty.pt"),
    "v6 SYNTH-500 (fill)":   os.path.join(ROOT, "models", "backups", "board_seg_v6_synth500.pt"),
}


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


def img_for(stem):
    for ext in (".png", ".jpg", ".jpeg", ".PNG"):
        p = os.path.join(BENCH, "images", stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    gt = parse_gt(os.path.join(BENCH, "ground_truth.txt"))
    items = [(s, img_for(s), g) for s, g in gt.items() if img_for(s)]
    straight = [it for it in items if it[0] not in SKEWED]
    skew = [it for it in items if it[0] in SKEWED]
    print(f"straight={len(straight)}  skewed={len(skew)}")

    backup = PROD + ".cmpbak"
    shutil.copy(PROD, backup)
    from boarddetection.pipeline import XiangqiRecognizer
    try:
        print(f"\n{'model':24s} | straight        | skewed")
        print("-" * 60)
        for name, path in MODELS.items():
            shutil.copy(path, PROD)
            rec = XiangqiRecognizer()           # reloads board_seg.pt fresh
            se = sum(is_exact(rec.recognize(ip).fen.split()[0], g)
                     for _, ip, g in straight)
            ke = sum(is_exact(rec.recognize(ip).fen.split()[0], g)
                     for _, ip, g in skew)
            print(f"{name:24s} | {se:3d}/{len(straight)} ({100*se/len(straight):4.1f}%) "
                  f"| {ke:2d}/{len(skew)} ({100*ke/len(skew):4.1f}%)")
    finally:
        shutil.copy(backup, PROD)
        os.remove(backup)
        print("\n(restored prod board_seg.pt)")


if __name__ == "__main__":
    main()
