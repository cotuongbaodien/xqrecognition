"""Detailed piece-error report on test/bench (deployed config).
For each board, aligns to the best-matching GT orientation (mirror-tolerant),
then aggregates per-cell mismatches into:
  MISSING  (GT has piece, detected empty)   -> per piece type
  EXTRA    (detected piece, GT empty)        -> per piece type
  WRONG    (both occupied, wrong type)       -> per (gt -> det) confusion pair
Also writes a per-board detail file.
"""
import sys, glob
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from boarddetection.pipeline import XiangqiRecognizer
BENCH = ROOT / "test/bench"

NAME = {  # FEN char -> readable
 'k':'k.tướng','a':'k.sĩ','b':'k.tượng','n':'k.mã','r':'k.xe','c':'k.pháo','p':'k.tốt',
 'K':'đ.tướng','A':'đ.sĩ','B':'đ.tượng','N':'đ.mã','R':'đ.xe','C':'đ.pháo','P':'đ.tốt'}


def parse_gt(p):
    g={}
    for ln in open(p,encoding='utf-8'):
        if ':' in ln and not ln.strip().startswith('#'):
            k,v=ln.split(':',1)
            if v.strip(): g[k.strip()]=v.strip()
    return g


def exp(f):
    rows=[]
    for row in f.split()[0].split('/'):
        c=[]
        for ch in row: c+=['.']*int(ch) if ch.isdigit() else [ch]
        rows.append((c+['.']*9)[:9])
    while len(rows)<10: rows.append(['.']*9)
    return rows[:10]


def mir(f): return '/'.join(''.join(r[::-1]) for r in exp(f))


def diff_cells(det, gt):
    """best-orientation per-cell mismatches: list of (kind, gt, det)."""
    best=None
    for g in (gt, mir(gt)):
        ra,rb=exp(det),exp(g); cells=[]; score=0
        for i in range(10):
            for j in range(9):
                a,b=ra[i][j],rb[i][j]
                if a=='.' and b!='.': cells.append(('MISS',b,'.')); score+=1
                elif a!='.' and b=='.': cells.append(('EXTRA','.',a)); score+=1
                elif a!='.' and b!='.' and a!=b: cells.append(('WRONG',b,a)); score+=1
        if best is None or score<best[0]: best=(score,cells)
    return best[1]


def img_for(s):
    for e in ('.png','.jpg','.jpeg'):
        p=BENCH/'images'/f'{s}{e}'
        if p.exists(): return str(p)


def main():
    gt=parse_gt(BENCH/'ground_truth.txt')
    rec=XiangqiRecognizer()
    miss=Counter(); extra=Counter(); conf=Counter()
    detail=open(BENCH/'piece_errors.txt','w',encoding='utf-8')
    for s in sorted(gt,key=lambda x:int(x) if x.isdigit() else 1e9):
        ip=img_for(s)
        if not ip: continue
        det=rec.recognize(ip).fen.split()[0]
        cells=diff_cells(det,gt[s])
        if not cells: continue
        parts=[]
        for kind,g,d in cells:
            if kind=='MISS': miss[g]+=1; parts.append(f"thiếu {NAME.get(g,g)}")
            elif kind=='EXTRA': extra[d]+=1; parts.append(f"dư {NAME.get(d,d)}")
            else: conf[(g,d)]+=1; parts.append(f"{NAME.get(g,g)}->{NAME.get(d,d)}")
        detail.write(f"{s}: "+"; ".join(parts)+"\n")
    detail.close()
    def show(title,cnt):
        print(f"\n{title} (tong {sum(cnt.values())}):")
        for k,v in cnt.most_common(12):
            lab=f"{NAME.get(k[0],k[0])}->{NAME.get(k[1],k[1])}" if isinstance(k,tuple) else NAME.get(k,k)
            print(f"  {v:3d}  {lab}")
    show("THIEU QUAN (missing)",miss)
    show("DU QUAN (extra)",extra)
    show("SAI LOAI (confusion gt->det)",conf)
    print(f"\nchi tiet tung ban -> {BENCH/'piece_errors.txt'}")


if __name__ == "__main__":
    main()
