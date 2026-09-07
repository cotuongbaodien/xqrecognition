"""Apply per-gallery label corrections from review_gallery.py sheets.

The user reads a gallery sheet and reports which tile #idx are mislabeled and
their true class. This maps each idx -> exact (label file, line) via
data/label_review/manifest_<tag>.json and rewrites the class id in place.
Only tiles whose current class still equals the gallery's class are touched
(idempotent / safe against double-apply).

Usage:
  python scripts/apply_review.py <tag> '<idx>=<class> <idx,idx>=<class> ...'
e.g. python scripts/apply_review.py xeden '0=tot 1,2,3,7=ma 72=phao'

class tokens (color inferred from the gallery's color unless given):
  ma=horse xe=chariot phao=cannon si=advisor tuong=elephant soai=general tot=soldier
  optional explicit color: ma-do, xe-den ...
"""
import json
import os
import sys
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# gallery tag -> (class id, color)  color: 'den'(black) / 'do'(red)
TAG2ID = {
    "xeden": (2, "den"), "xedo": (13, "do"),
    "maden": (5, "den"), "mado": (16, "do"),
    "phaoden": (1, "den"), "phaodo": (12, "do"),
    "siden": (0, "den"), "sido": (11, "do"),
    "tuongden": (3, "den"), "tuongdo": (14, "do"),   # tuong = elephant
    "soaiden": (4, "den"), "soaido": (15, "do"),     # soai = general
    "totden": (6, "den"), "totdo": (17, "do"),
}
# (piece, color) -> class id
PIECE = {
    ("xe", "den"): 2, ("xe", "do"): 13,
    ("ma", "den"): 5, ("ma", "do"): 16,
    ("phao", "den"): 1, ("phao", "do"): 12,
    ("si", "den"): 0, ("si", "do"): 11,
    ("tuong", "den"): 3, ("tuong", "do"): 14,   # elephant
    ("soai", "den"): 4, ("soai", "do"): 15,     # general (tướng/将帥)
    ("vua", "den"): 4, ("vua", "do"): 15,        # alias: VUA(king) — tránh lẫn "tượng"
    ("tot", "den"): 6, ("tot", "do"): 17,
}
NAME = {v: k for k, v in {
    "black-advisor": 0, "black-cannon": 1, "black-chariot": 2,
    "black-elephant": 3, "black-general": 4, "black-horse": 5,
    "black-soldier": 6, "red-advisor": 11, "red-cannon": 12,
    "red-chariot": 13, "red-elephant": 14, "red-general": 15,
    "red-horse": 16, "red-soldier": 17}.items()}
NAME[99] = "DELETE(sentinel)"


def parse_class(tok, gallery_color):
    tok = tok.strip().lower()
    if tok in ("bo", "del", "xoa", "99"):
        return 99          # sentinel: mark for deletion (purged at the end)
    color = gallery_color
    for c in ("den", "do"):
        if tok.endswith("-" + c) or tok.endswith(c):
            if tok.endswith("-" + c):
                tok = tok[:-(len(c) + 1)]
                color = c
            elif tok in ("den", "do"):
                pass
    base = tok.split("-")[0]
    return PIECE[(base, color)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("spec")
    ap.add_argument("--from-any", action="store_true",
                    help="also change cells that were ALREADY re-labelled "
                         "(use to undo a wrong fix; without it they are skipped)")
    ap.add_argument("--dir", default="data/label_review",
                    help="review dir holding manifest_<tag>.json "
                         "(e.g. data/label_review_incoming for weekly ingest)")
    args = ap.parse_args()
    tag, spec = args.tag, args.spec
    old, gcolor = TAG2ID[tag]
    man = {e["idx"]: e for e in json.load(
        open(f"{ROOT}/{args.dir}/manifest_{tag}.json"))}

    truth = {}
    for grp in spec.split():
        idxs, cls = grp.split("=")
        cid = parse_class(cls, gcolor)
        for i in idxs.split(","):
            truth[int(i)] = cid

    byfile = defaultdict(dict)
    chg = Counter()
    for idx, new in truth.items():
        if new == old and not args.from_any:
            continue
        e = man[idx]
        byfile[e["file"]][e["line"]] = new
    nl, skip, missing = 0, 0, 0
    for f, lines in byfile.items():
        path = os.path.join(ROOT, f)
        if not os.path.exists(path):
            # source image was de-leaked/quarantined (held-out) — label gone,
            # nothing to fix (it won't be trained on). Skip gracefully.
            missing += len(lines)
            continue
        L = open(path, encoding="utf-8").read().splitlines()
        for li, new in lines.items():
            p = L[li].split()
            was = int(p[0])
            # normally only cells still holding this gallery's class may change:
            # a second report of the same idx is a typo, not a re-fix. --from-any
            # lifts that so an earlier WRONG fix can be corrected.
            if was != old and not args.from_any:
                skip += 1
                continue
            if was == new:
                continue
            p[0] = str(new)
            L[li] = " ".join(p)
            chg[(NAME[was], NAME[new])] += 1
            nl += 1
        open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"{tag}: sua {nl} box (skip {skip}, de-leaked/missing {missing})")
    for k, v in chg.most_common():
        print(f"  {v:3d}  {k[0]} -> {k[1]}")
    _proof(tag, truth, man, old, os.path.join(ROOT, args.dir))


def _proof(tag, truth, man, old, review_dir):
    """Write a contact sheet of the cells just changed, each captioned with its
    new class.

    On 2026-08-17 a whole class (172 cells) was applied against the wrong
    gallery: the reviewer was reading one review dir while --dir pointed at
    another, and the two number their idx differently. Nothing in the output
    revealed it — the counts looked perfectly normal. One glance at these crops
    does: if the pieces do not match the captions, the --dir is wrong. Cheap
    insurance, so it runs on every apply.
    """
    import cv2
    import numpy as np

    picks = [(i, n) for i, n in truth.items() if n != old][:24]
    if not picks:
        return
    stage = os.path.dirname(os.path.dirname(os.path.join(ROOT, man[picks[0][0]]["file"])))
    sz, cols = 110, 8
    rows_n = (len(picks) + cols - 1) // cols
    canvas = np.full((rows_n * (sz + 20), cols * sz, 3), 255, np.uint8)
    for k, (idx, new) in enumerate(picks):
        e = man[idx]
        ip = os.path.join(stage, "images", e["src"])
        im = cv2.imdecode(np.fromfile(ip, np.uint8), cv2.IMREAD_COLOR) \
            if os.path.exists(ip) else None
        if im is None:
            continue
        lp = os.path.join(ROOT, e["file"])
        p = open(lp, encoding="utf-8").read().splitlines()[e["line"]].split()
        H, W = im.shape[:2]
        x, y, w, h = [float(v) for v in p[1:5]]
        cr = im[max(0, int((y - h / 2) * H)):int((y + h / 2) * H),
                max(0, int((x - w / 2) * W)):int((x + w / 2) * W)]
        if cr.size == 0:
            continue
        r0, c0 = (k // cols) * (sz + 20), (k % cols) * sz
        canvas[r0:r0 + sz, c0:c0 + sz] = cv2.resize(cr, (sz, sz))
        cv2.putText(canvas, f"{NAME[new]} #{idx}", (c0 + 2, r0 + sz + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 255), 1)
    out = os.path.join(review_dir, f"_applied_{tag}.jpg")
    cv2.imencode(".jpg", canvas)[1].tofile(out)
    print(f"  kiem chung: {os.path.relpath(out, ROOT)} "
          f"(quan phai khop nhan; khong khop = sai --dir)")


if __name__ == "__main__":
    main()
