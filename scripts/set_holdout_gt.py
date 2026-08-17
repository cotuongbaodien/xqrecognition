"""Record human-verified FEN for the bench_holdout images (244-943).

The draft FENs in prodfen_draft.txt come from the deployed model, so they are
NOT ground truth — a draft only becomes GT once a human has looked at the image
and said so. This script is the only writer of bench_holdout_gt.txt, and it
refuses anything that is not a legal 10x9 board, so a typo can never turn into
a silently-wrong benchmark answer.

  python scripts/set_holdout_gt.py "943: 2rakab2/1C7/... w - - 0 1"
  python scripts/set_holdout_gt.py "244 ok" "245 ok" "246 bo"

  <num>: <fen>   human-supplied FEN (wins over the draft)
  <num> ok       the draft line is correct -> copy it into GT
  <num> bo       unusable image (not a board / cropped) -> drop from the bench
"""
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = os.path.join(ROOT, "ingest", "2026-07-11")
IMAGES = os.path.join(BATCH, "bench_holdout")
DRAFT = os.path.join(BATCH, "prodfen_draft.txt")
GT = os.path.join(BATCH, "bench_holdout_gt.txt")
DROPPED = os.path.join(BATCH, "bench_holdout_dropped.txt")
# Where the human actually browses the images from (a copy, not the master).
WORKING_COPY = os.path.join(
    os.path.expanduser("~"), "Downloads", "bench_holdout")

GT_HEADER = (
    "# GROUND TRUTH da duoc NGUOI xac nhan, cho bench_holdout/<num>.jpg (244-943).\n"
    "# Chi ghi dong nao da mat thay. Day moi la nguon promote sang test/bench.\n"
)
MAX_COUNTS = {"k": 1, "K": 1, "a": 2, "A": 2, "b": 2, "B": 2,
              "n": 2, "N": 2, "r": 2, "R": 2, "c": 2, "C": 2, "p": 5, "P": 5}


def read_entries(path):
    """num -> line value, keeping '' for blank draft lines."""
    out = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            if ln.startswith("#") or ":" not in ln:
                continue
            k, v = ln.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def check(fen):
    """Return (normalised_fen, error). Board part must be a legal 10x9 grid."""
    board = fen.split()[0]
    ranks = board.split("/")
    if len(ranks) != 10:
        return None, f"{len(ranks)} hang (can 10)"
    for i, r in enumerate(ranks):
        if not re.fullmatch(r"[1-9rnbakcpRNBAKCP]+", r):
            return None, f"hang {i + 1} co ky tu la: {r}"
        n = sum(int(ch) if ch.isdigit() else 1 for ch in r)
        if n != 9:
            return None, f"hang {i + 1} = {n} cot (can 9): {r}"
    counts = Counter(ch for ch in board if ch.isalpha())
    for sym, mx in MAX_COUNTS.items():
        if counts[sym] > mx:
            return None, f"co {counts[sym]} quan '{sym}' (toi da {mx})"
    warn = [s for s in ("k", "K") if counts[s] == 0]
    if len(fen.split()) == 1:
        fen += " w - - 0 1"
    return fen, ("THIEU tuong " + "/".join(warn)) if warn else None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    draft, gt = read_entries(DRAFT), read_entries(GT)
    dropped = set(read_entries(DROPPED))
    ok = bad = 0
    for arg in sys.argv[1:]:
        for tok in [t.strip() for t in arg.split(",") if t.strip()]:
            m = re.match(r"^(\d{1,3})\s*(?::\s*(.+)|\s+(ok|bo|del))$", tok, re.I)
            if not m:
                print(f"  ?? khong hieu: {tok!r}")
                bad += 1
                continue
            num = f"{int(m.group(1)):03d}"
            if not os.path.exists(os.path.join(IMAGES, num + ".jpg")):
                print(f"  {num} KHONG co anh -> bo qua")
                bad += 1
                continue
            word = (m.group(3) or "").lower()
            if word in ("bo", "del"):
                dropped.add(num)
                gt.pop(num, None)
                print(f"  {num} -> LOAI khoi bench")
                ok += 1
                continue
            fen = m.group(2) if m.group(2) else draft.get(num, "")
            if not fen:
                print(f"  {num} 'ok' nhung dong nhap TRONG -> phai go tay")
                bad += 1
                continue
            fen, err = check(fen)
            if fen is None:
                print(f"  {num} SAI DINH DANG: {err}")
                bad += 1
                continue
            if err:
                print(f"  {num} !! canh bao: {err}")
            was = gt.get(num)
            gt[num] = fen
            dropped.discard(num)
            same = " (giong nhap)" if m.group(3) else \
                   (" (khac nhap)" if draft.get(num) and
                    fen.split()[0] != draft[num].split()[0] else "")
            print(f"  {num} {'sua lai' if was else 'ghi'}{same}")
            ok += 1

    with open(GT, "w", encoding="utf-8") as f:
        f.write(GT_HEADER)
        for k in sorted(gt):
            f.write(f"{k}: {gt[k]}\n")
    with open(DROPPED, "w", encoding="utf-8") as f:
        f.write("# Anh KHONG dung lam bench (khong phai ban co / chup thieu).\n")
        for k in sorted(dropped):
            f.write(f"{k}: loai\n")

    # Keep the human's working copy in sync: an image whose FEN is settled is
    # removed there so it never gets opened twice. Only ever deletes from the
    # working copy, and only while the master under bench_holdout/ still holds
    # the original.
    if os.path.isdir(WORKING_COPY):
        gone = [n for n in sorted(set(gt) | dropped)
                if os.path.exists(os.path.join(IMAGES, n + ".jpg"))
                and os.path.exists(os.path.join(WORKING_COPY, n + ".jpg"))]
        for n in gone:
            os.remove(os.path.join(WORKING_COPY, n + ".jpg"))
        if gone:
            print(f"  xoa khoi ban lam viec: {', '.join(gone)}")

    total = len([n for n in os.listdir(IMAGES) if n.endswith(".jpg")])
    done = len(gt) + len(dropped)
    print(f"\nghi {ok}, loi {bad} | da xong {done}/{total} "
          f"({len(gt)} co FEN, {len(dropped)} loai) | con {total - done}")


if __name__ == "__main__":
    main()
