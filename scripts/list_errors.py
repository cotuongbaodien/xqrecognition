"""List per-board error type OLD-prod vs NEW (v5_640), to compare what each
gets wrong. Splits errors into GRID (occupancy: piece present/empty mismatch =
board localization) vs TYPE (right cell, wrong piece class = items model).
Mirror-tolerant.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from boarddetection.pipeline import XiangqiRecognizer
from boarddetection.board_segmenter import BoardSegmenter

TEST = ROOT / "test"
OLD = "models/backups/board_seg_predeploy_2026-06-16.pt"
NEW = "boarddetection/models/board_seg.pt"


def gt_map():
    g = {}
    for ln in open(TEST / "ground_truth.txt", encoding="utf-8"):
        if ":" in ln:
            k, v = ln.split(":", 1); g[k.strip()] = v.strip().split()[0]
    return g


def exp(f):
    r = []
    for row in f.split("/"):
        c = []
        for ch in row:
            c += ["."] * int(ch) if ch.isdigit() else [ch]
        r.append((c + ["."] * 9)[:9])
    while len(r) < 10: r.append(["."] * 9)
    return r[:10]


def mir(f): return "/".join("".join(r[::-1]) for r in exp(f))


def errs(det, gt):
    """(grid_err, type_err) vs the better of gt / mirror(gt)."""
    best = None
    for g in (gt, mir(gt)):
        ra, rb = exp(det), exp(g)
        grid = typ = 0
        for i in range(10):
            for j in range(9):
                a, b = ra[i][j], rb[i][j]
                if (a != ".") != (b != "."): grid += 1
                elif b != "." and a != b: typ += 1
        if best is None or (grid + typ) < sum(best): best = (grid, typ)
    return best


def run(rec, mp, imgs):
    rec.board_segmenter = BoardSegmenter(mp)
    return {i.stem: rec.recognize(str(i)).fen.split()[0] for i in imgs}


def main():
    gt = gt_map()
    imgs = sorted([f for f in TEST.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
                  key=lambda p: int(p.stem) if p.stem.isdigit() else 1e9)
    rec = XiangqiRecognizer()
    old = run(rec, OLD, imgs); new = run(rec, NEW, imgs)
    print(f"{'img':4s} | {'OLD prod':22s} | {'NEW v5_640':22s} | note")
    print("-" * 78)
    for k in sorted(gt, key=lambda x: int(x) if x.isdigit() else 1e9):
        go, gn = errs(old.get(k, ""), gt[k]), errs(new.get(k, ""), gt[k])
        ok_o, ok_n = sum(go) == 0, sum(gn) == 0
        if ok_o and ok_n:
            continue
        def fmt(e, ok):
            if ok: return "OK"
            return f"grid={e[0]} piece={e[1]}"
        note = ""
        if ok_o and not ok_n: note = "<< REGRESS"
        elif ok_n and not ok_o: note = "improved"
        elif gn[0] > go[0]: note = "grid worse"
        elif gn[0] < go[0]: note = "grid better"
        print(f"{k:4s} | {fmt(go, ok_o):22s} | {fmt(gn, ok_n):22s} | {note}")
    print("-" * 78)
    print(f"OLD exact: {sum(1 for k in gt if sum(errs(old.get(k,''),gt[k]))==0)}/{len(gt)}  "
          f"NEW exact: {sum(1 for k in gt if sum(errs(new.get(k,''),gt[k]))==0)}/{len(gt)}")


if __name__ == "__main__":
    main()
