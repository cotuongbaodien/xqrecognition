"""Mirror-tolerant FEN evaluation against ground truth.

Compares detected FEN (from a detection output results.json) with
ground-truth FEN. Xiangqi boards are left-right symmetric and the
pipeline does NOT disambiguate horizontal mirror, so a detection is
counted correct if it matches the GT OR the GT's horizontal mirror.

Also separates grid errors (occupancy mismatch — piece in wrong cell or
missing/extra) from pure piece-type errors (right cell, wrong piece).

Usage:
    python scripts/eval_fen.py --gt test/ground_truth.txt --results test/output_v9/results.json
"""

import argparse
import json
import os
import re


def parse_gt(path):
    gt = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, fen = line.split(":", 1)
        gt[key.strip()] = fen.strip().split()[0]
    return gt


def expand_rows(fen):
    rows = []
    for row in fen.split("/"):
        cells = []
        for ch in row:
            if ch.isdigit():
                cells += ["."] * int(ch)
            else:
                cells.append(ch)
        cells = (cells + ["."] * 9)[:9]
        rows.append(cells)
    while len(rows) < 10:
        rows.append(["."] * 9)
    return rows[:10]


def mirror_fen(fen):
    rows = expand_rows(fen)
    mrows = []
    for r in rows:
        rr = r[::-1]
        # recompress
        out = ""
        cnt = 0
        for c in rr:
            if c == ".":
                cnt += 1
            else:
                if cnt:
                    out += str(cnt); cnt = 0
                out += c
        if cnt:
            out += str(cnt)
        mrows.append(out or "9")
    return "/".join(mrows)


def diff(a, b):
    """Return (rows_differ, type_err, missing_extra) comparing two FENs."""
    ra, rb = expand_rows(a), expand_rows(b)
    rows_d = sum(1 for i in range(10) if ra[i] != rb[i])
    type_err = miss = 0
    for i in range(10):
        for j in range(9):
            ga, gb = rb[i][j], ra[i][j]  # rb=truth, ra=detected
            if (ga != ".") == (gb != "."):
                if ga != "." and ga != gb:
                    type_err += 1
            else:
                miss += 1
    return rows_d, type_err, miss


def stem_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="test/ground_truth.txt")
    ap.add_argument("--results", default="test/output_v9/results.json")
    args = ap.parse_args()

    gt = parse_gt(args.gt)
    results = json.load(open(args.results))
    det = {stem_of(r["image"]): r["fen"].split()[0] for r in results}

    exact = 0
    total_wrong = 0
    n = 0
    print(f"{'img':5s} {'status':14s} {'mirror?':8s} grid/piece")
    print("-" * 50)
    for key in sorted(gt, key=lambda k: int(k) if k.isdigit() else 1e9):
        if key not in det:
            print(f"{key:5s} NO DETECTION")
            continue
        n += 1
        d = det[key]
        g = gt[key]
        gm = mirror_fen(g)
        # direct vs mirror — pick better
        dd = diff(d, g)
        dm = diff(d, gm)
        use_mirror = (dm[0] < dd[0])
        best = dm if use_mirror else dd
        rows_d, type_err, miss = best
        if rows_d == 0:
            exact += 1
            status = "EXACT"
        else:
            status = f"{rows_d}/10 rows"
        total_wrong += rows_d
        mflag = "mirror" if use_mirror else ""
        grid_note = f"type={type_err} miss/extra={miss}"
        print(f"{key:5s} {status:14s} {mflag:8s} {grid_note}")

    print("-" * 50)
    print(f"EXACT (mirror-tolerant): {exact}/{n}")
    print(f"Total wrong rows: {total_wrong}")
    print(f"\nGrid OK if miss/extra≈0 and errors are type-only "
          f"(piece classification, not grid).")


if __name__ == "__main__":
    main()
