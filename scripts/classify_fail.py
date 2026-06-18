"""Classify each WRONG board on test/bench (deployed config) as GRID-suspect
vs PIECE-only. Heuristic: a mislocalized grid scatters pieces into wrong cells
-> high (missing+extra). A pure class error keeps cells but wrong piece type
-> high wrong-class, low missing+extra.
GRID-suspect if (missing+extra) >= 3 AND (missing+extra) >= wrong_class.
"""
import sys, glob
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from boarddetection.pipeline import XiangqiRecognizer
BENCH = ROOT / "test/bench"


def parse_gt(p):
    g = {}
    for ln in open(p, encoding="utf-8"):
        if ":" in ln and not ln.strip().startswith("#"):
            k, v = ln.split(":", 1)
            if v.strip(): g[k.strip()] = v.strip()
    return g


def exp(f):
    rows = []
    for row in f.split()[0].split("/"):
        c = []
        for ch in row: c += ["."]*int(ch) if ch.isdigit() else [ch]
        rows.append((c+["."]*9)[:9])
    while len(rows) < 10: rows.append(["."]*9)
    return rows[:10]


def mir(f): return "/".join("".join(r[::-1]) for r in exp(f))


def err(det, gt):
    best = None
    for g in (gt, mir(gt)):
        ra, rb = exp(det), exp(g); mi=ex=wr=0
        for i in range(10):
            for j in range(9):
                a,b=ra[i][j],rb[i][j]
                if a=="." and b!=".": mi+=1
                elif a!="." and b==".": ex+=1
                elif a!="." and b!="." and a!=b: wr+=1
        if best is None or (mi+ex+wr)<sum(best): best=(mi,ex,wr)
    return best


def img_for(s):
    for e in (".png",".jpg",".jpeg"):
        p=BENCH/"images"/f"{s}{e}"
        if p.exists(): return str(p)


def main():
    gt = parse_gt(BENCH/"ground_truth.txt")
    rec = XiangqiRecognizer()
    grid_fail, piece_fail = [], []
    for s in sorted(gt, key=lambda x:int(x) if x.isdigit() else 1e9):
        ip = img_for(s)
        if not ip: continue
        det = rec.recognize(ip).fen.split()[0]
        mi,ex,wr = err(det, gt[s])
        if mi+ex+wr == 0: continue
        if (mi+ex) >= 3 and (mi+ex) >= wr:
            grid_fail.append((s,mi,ex,wr))
        else:
            piece_fail.append((s,mi,ex,wr))
    print(f"GRID-suspect ({len(grid_fail)}):")
    print("  " + ", ".join(s for s,_,_,_ in grid_fail))
    print(f"\nPIECE-only ({len(piece_fail)}):")
    print("  " + ", ".join(s for s,_,_,_ in piece_fail))
    with open(BENCH/"grid_fail.txt","w",encoding="utf-8") as w:
        for s,_,_,_ in grid_fail: w.write(s+"\n")
    print(f"\n-> grid-suspect list saved to {BENCH/'grid_fail.txt'}")


if __name__ == "__main__":
    main()
