"""Compare two ITEMS models on the 86-image test set (mirror-tolerant FEN).
Same board_seg for both runs (isolates the items-model effect)."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from boarddetection.pipeline import XiangqiRecognizer

TEST = ROOT / "test"


def parse_gt(p):
    g = {}
    for ln in open(p, encoding="utf-8"):
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
def ok(d, g): return exp(d) == exp(g) or exp(d) == exp(mir(g))


def run(rec, mp, imgs):
    rec.item_detector.load_model(mp)
    return {i.stem: rec.recognize(str(i)).fen.split()[0] for i in imgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="models/backups/items_v16_960.pt")
    ap.add_argument("--baseline", default="boarddetection/models/items.pt")
    a = ap.parse_args()
    gt = parse_gt(TEST / "ground_truth.txt")
    imgs = sorted([f for f in TEST.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")],
                  key=lambda p: int(p.stem) if p.stem.isdigit() else 1e9)
    rec = XiangqiRecognizer()
    print("baseline..."); base = run(rec, a.baseline, imgs)
    print("candidate..."); cand = run(rec, a.candidate, imgs)
    def score(d): return sum(1 for k in gt if k in d and ok(d[k], gt[k]))
    bs, cs = score(base), score(cand)
    print(f"\nbaseline items.pt : {bs}/{len(gt)}")
    print(f"candidate v16_960 : {cs}/{len(gt)}")
    print("\nCHANGES:")
    order = lambda k: int(k) if k.isdigit() else 1e9
    for k in sorted(gt, key=order):
        bo = k in base and ok(base[k], gt[k]); co = k in cand and ok(cand[k], gt[k])
        if bo != co:
            print(f"  {k:4s} {'OK' if bo else 'wrong':6s} -> {'OK' if co else 'wrong':6s} {'IMPROVE' if co else 'REGRESS'}")


if __name__ == "__main__":
    main()
