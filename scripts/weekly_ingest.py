"""Weekly OCR-submission ingest -> pseudo-label -> staging -> review gallery.

Active-learning loop for growing items_v20 from real user submissions:

  1. v20 `items.pt` detects on a week's submitted images (conf>=0.25, imgsz=960,
     matching deploy inference). Pseudo-labels written in YOLO format.
  2. Images + labels staged into  items_v20/incoming/{images,labels}/  (NOT
     train/ yet) so wrong pseudo-labels never pollute the trainset before review.
  3. Per-class review galleries built into  data/label_review_incoming/  in the
     SAME format as review_gallery.py, so the exact same fix workflow applies:
         python scripts/apply_review.py xeden '5,12=ma' --dir data/label_review_incoming
  4. After review, finalize:
         python scripts/weekly_ingest.py --merge
     -> purges sentinel-99 ("bo") lines, moves incoming/* into train/.

Usage:
  # ingest one week's folder (run when GPU is free; detect is heavy):
  python scripts/weekly_ingest.py --input data/incoming/2026-06-24
  # ... review + fix via apply_review.py --dir data/label_review_incoming ...
  python scripts/weekly_ingest.py --merge          # move into train/, then retrain
"""
import argparse
import glob
import os
import shutil
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boarddetection.settings import ITEM_CLASSES, ITEMS_MODEL  # noqa: E402

# Self-contained, PER-PERIOD workspace (gitignored). One isolated folder per
# ingest period (run ~every 1-2 months, not weekly). Point --batch at it:
#   ingest/<period>/inbox/         <- raw images to pseudo-label
#   ingest/<period>/bench_holdout/ <- images held out for the FEN benchmark
#   ingest/<period>/staging/{images,labels} <- pseudo-labeled, pre-review (auto)
#   ingest/<period>/review/<N.tag>/sheet_*  <- label review galleries (auto)
# On --merge, that period's staging/ moves into data/items_v20/train/.
# WEEKLY/INBOX/STAGE/REVIEW_DIR below are DEFAULTS; --batch overrides them in
# main() so every path is scoped under the chosen period folder.
WEEKLY = os.path.join(ROOT, "weekly")
INBOX = os.path.join(WEEKLY, "inbox")
STAGE = os.path.join(WEEKLY, "staging")
REVIEW_DIR = os.path.join(WEEKLY, "review")
DATASET = os.path.join(ROOT, "data", "items_v20")   # canonical trainset (merge target)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# class id -> gallery tag (only the 14 pieces are reviewed; landmarks 7-10 are
# pseudo-labeled and kept for training but not surfaced in galleries).
CID2TAG = {
    2: "xeden", 13: "xedo", 5: "maden", 16: "mado", 1: "phaoden", 12: "phaodo",
    0: "siden", 11: "sido", 3: "tuongden", 14: "tuongdo", 4: "soaiden",
    15: "soaido", 6: "totden", 17: "totdo",
}
ORDER = ["xeden", "maden", "xedo", "mado", "phaoden", "phaodo", "tuongden",
         "tuongdo", "totden", "totdo", "siden", "sido", "soaiden", "soaido"]
PREFIX = {tag: i + 1 for i, tag in enumerate(ORDER)}
SZ, COLS, ROWS = 80, 10, 10
PER = COLS * ROWS

# How many of each piece a legal board can hold. A picture pseudo-labelled with
# MORE than this many of one class must contain a mistake — the strongest cheap
# signal we have for ranking cells by suspicion (see build_galleries).
MAXN = {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 2, 6: 5,          # black
        11: 2, 12: 2, 13: 2, 14: 2, 15: 1, 16: 2, 17: 5}   # red
PALACE_CIDS = {0, 11, 4, 15}      # advisors + generals: must sit in the palace
PALACE_LANDMARKS = {8, 9, 10}     # palace-bottom / palace-center / palace-conner


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _week_of(path, input_dir):
    """Derive the week tag from an image path. Handles both a flat folder and a
    parent dir of week-subfolders (e.g. ocrservice/2026-W24/images/x.jpg)."""
    rel = os.path.relpath(path, input_dir).replace("\\", "/").split("/")
    if len(rel) > 1:
        return rel[0]                                   # week subfolder name
    return os.path.basename(os.path.normpath(input_dir))


def detect_and_stage(input_dir, conf, model_path=None, device=None, period=None):
    from ultralytics import YOLO
    # `period` (when set) is used as the staged-name tag for ALL images, so a
    # flat inbox stages as weekly_<period>_<stem> instead of weekly_inbox_<stem>.
    wk = lambda p: period if period else _week_of(p, input_dir)
    # recursive: --input can be ONE week folder OR a parent of week-subfolders
    # (each with images/). Skip anything under a labels/ dir.
    imgs = [p for p in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
            if p.lower().endswith(IMG_EXT)
            and (os.sep + "labels" + os.sep) not in p]
    if not imgs:
        print(f"No images in {input_dir}")
        return None
    weeks = sorted({wk(p) for p in imgs})
    print(f"Weeks: {', '.join(weeks)}")
    os.makedirs(os.path.join(STAGE, "images"), exist_ok=True)
    os.makedirs(os.path.join(STAGE, "labels"), exist_ok=True)

    mpath = model_path or str(ITEMS_MODEL)
    print(f"Loading {mpath} ...")
    model = YOLO(mpath)
    # ONNX weights load via onnxruntime; if only CPUExecutionProvider is present
    # (no onnxruntime-gpu), inference MUST run on cpu or ultralytics errors on
    # GPU IO-binding. .pt weights default to auto-GPU when device is None.
    dev = device
    if dev is None and mpath.lower().endswith(".onnx"):
        dev = "cpu"
    print(f"Detecting {len(imgs)} images (conf>={conf}, imgsz=960, device={dev or 'auto'}) ...")

    dist = Counter()
    n_imgs = n_box = 0
    for ip in imgs:
        im = imread_u(ip)
        if im is None:
            print(f"  skip unreadable: {ip}")
            continue
        H, W = im.shape[:2]
        res = model(im, conf=conf, imgsz=960, verbose=False, device=dev)[0]
        lines = []
        if res.boxes is not None:
            for i in range(len(res.boxes)):
                cid = int(res.boxes.cls[i].cpu().numpy())
                if cid not in ITEM_CLASSES:
                    continue
                x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().tolist()
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                dist[cid] += 1
        # unique staged name: weekly_<week>_<origstem>
        stem = os.path.splitext(os.path.basename(ip))[0]
        base = f"weekly_{wk(ip)}_{stem}"
        ext = os.path.splitext(ip)[1].lower()
        shutil.copy(ip, os.path.join(STAGE, "images", base + ext))
        with open(os.path.join(STAGE, "labels", base + ".txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_imgs += 1
        n_box += len(lines)

    print(f"\nStaged {n_imgs} images, {n_box} boxes -> {STAGE}")
    print("Class distribution (pseudo-labels):")
    for cid, c in dist.most_common():
        print(f"  {c:5d}  {ITEM_CLASSES[cid][0]}")
    return n_imgs


def _suspicion(items, cid):
    """Rank a class's cells worst-first. Three signals, in priority order:

    1. **excess** — how many boxes of this class the picture holds beyond what
       a legal board allows. A photo with 3 "black general" boxes has at least
       two wrong ones; a photo with 1 is almost certainly right. Measured on
       the 150 hand-found errors in `soaiden` (2026-08-17): 0.29% of cells in
       1-box pictures were wrong vs 18% / 43% / 70% in 2- / 3- / 4-box ones.
    2. **size deviation** — |log(area / class median area)|. Catches the junk
       crops (half a piece, a board corner) that get marked `bo`.
    3. **palace distance** — generals and advisors may not leave the palace, so
       distance to the nearest palace landmark flags impossible placements.
       Meaningless for the other classes, which get 0.

    ⚠️ **DO NOT TRUST THIS — it is off by default for a reason.** The ranking
    was validated on `soaiden` (5% of the class -> 65% of its errors, 10% ->
    93%) and then failed on `soaido`, where the very same signal ran *backwards*:

        cells per picture     soaiden wrong     soaido wrong
        1                          0.29%            2.66%
        2                         18.4%             0.30%
        3                         43.5%             0.00%
        4+                        70.0%             0.00%

    A picture holding six red generals really does hold six red generals — it
    is a photo of a pile of pieces, and every crop in it is correctly labelled.
    Black generals are the ones the detector over-predicts, so there the extra
    boxes are the mistakes. Model confidence inverts between the two classes
    too. Which way a signal points is a property of the class, and there is no
    way to know it before a human has labelled that class.

    Reviewing `soaido`'s first 12 "hot" sheets caught 3 of its 172 errors —
    worse than reading the sheets in file order. Anyone reviving this must
    first show the correlation holds on at least two independently-labelled
    classes.
    """
    areas = np.array([it["area"] for it in items])
    med = np.median(areas) if len(areas) else 1.0
    for it in items:
        it["adev"] = abs(np.log(max(it["area"], 1e-9) / max(med, 1e-9)))
        if cid not in PALACE_CIDS:
            it["geo"] = 0.0
    items.sort(key=lambda it: (-it["exc"], -it["adev"], -it["geo"]))
    return items


def load_reviewed(path):
    """Read a `reviewed.json` progress file -> {(label file, line)} already seen.

    A review pass is recorded against the gallery it was done on, because idx
    only means anything there:

        {"manifest_dir": "ingest/2026-07-11/review_2026-08-17",
         "done": {"soaiden": "all",                  # whole class checked
                  "tuongden": {"max_idx": 3899},     # stopped after sheet 039
                  "xeden": {"src_prefix": ["weekly_2026-07-11_"]}}}

    Cells are keyed by (file, line), not by class, so a box the reviewer moved
    to another class stays skipped in its new class's gallery too.
    """
    import json
    cfg = json.load(open(path, encoding="utf-8"))
    mdir = os.path.join(ROOT, cfg["manifest_dir"])
    seen = set()
    for tag, rule in cfg.get("done", {}).items():
        mf = os.path.join(mdir, f"manifest_{tag}.json")
        if not os.path.exists(mf):
            print(f"  ! {tag}: no manifest in {cfg['manifest_dir']}, skipped")
            continue
        n = 0
        for e in json.load(open(mf, encoding="utf-8")):
            if rule == "all":
                ok = True
            elif "max_idx" in rule:
                ok = e["idx"] <= rule["max_idx"]
            else:
                ok = any(e["src"].startswith(x) for x in rule["src_prefix"])
            if ok:
                seen.add((e["file"], e["line"]))
                n += 1
        print(f"  {tag:8s}: {n} o da review -> bo qua")
    return seen


def build_galleries(review_dir=None, tile=None, sort_susp=False, skip=None):
    """Per-class review sheets + manifests for the staged incoming labels.
    Mirrors review_gallery.py output so apply_review.py works unchanged.

    Re-runnable: rebuilding after a review pass drops every box already
    re-labelled (it now belongs to another class's gallery) and every box
    marked for deletion (sentinel 99), so a fresh set only shows what is
    still labelled as that class.

    `sort_susp` orders each class worst-first (see _suspicion) instead of by
    filename, so the reviewer meets almost every mistake in the first sheets
    and can stop once they dry up.

    `skip` is a set of (label file, line) already reviewed in an earlier
    gallery; those cells are left out entirely, so a rebuild after adding
    new pictures shows ONLY what nobody has looked at yet."""
    import json
    review_dir = review_dir or REVIEW_DIR
    sz = tile or SZ
    # Pass 1 (labels only, no image decode): per-picture class counts, needed
    # for the excess signal before any cropping happens.
    counts = {}
    for lp in glob.glob(os.path.join(STAGE, "labels", "*.txt")):
        c = Counter()
        for ln in open(lp, encoding="utf-8"):
            p = ln.split()
            if p:
                c[int(p[0])] += 1
        counts[os.path.basename(lp)] = c
    per_class = defaultdict(list)   # cid -> [dict(crop, file, line, src, ...)]
    for ip in glob.glob(os.path.join(STAGE, "images", "*")):
        lp = os.path.join(STAGE, "labels",
                          os.path.splitext(os.path.basename(ip))[0] + ".txt")
        if not os.path.exists(lp):
            continue
        im = imread_u(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        rowsl = [ln.split() for ln in open(lp, encoding="utf-8")]
        cnt = counts.get(os.path.basename(lp), Counter())
        palace = [(float(p[1]), float(p[2])) for p in rowsl
                  if p and int(p[0]) in PALACE_LANDMARKS]
        for li, p in enumerate(rowsl):
            if not p:
                continue
            cid = int(p[0])
            if cid not in CID2TAG:
                continue
            rel = os.path.relpath(lp, ROOT).replace("\\", "/")
            if skip and (rel, li) in skip:
                continue
            x, yy, w, h = [float(v) for v in p[1:5]]
            x1, y1 = int((x - w / 2) * W), int((yy - h / 2) * H)
            x2, y2 = int((x + w / 2) * W), int((yy + h / 2) * H)
            cr = im[max(0, y1):y2, max(0, x1):x2]
            if cr.size == 0:
                continue
            per_class[cid].append({
                "crop": cv2.resize(cr, (sz, sz)), "file": rel, "line": li,
                "src": os.path.basename(ip),
                "exc": max(0, cnt[cid] - MAXN.get(cid, 99)),
                "area": w * h,
                "geo": (min(np.hypot(a - x, b - yy) for a, b in palace)
                        if palace else 9.0),
            })

    os.makedirs(review_dir, exist_ok=True)
    for cid, tag in CID2TAG.items():
        items = per_class.get(cid, [])
        if sort_susp and items:
            items = _suspicion(items, cid)
        d = os.path.join(review_dir, f"{PREFIX[tag]}.{tag}")
        os.makedirs(d, exist_ok=True)
        for old in glob.glob(os.path.join(d, "sheet_*.jpg")):
            os.remove(old)
        manifest = []
        nsheets = (len(items) + PER - 1) // PER
        for s in range(nsheets):
            chunk = items[s * PER:(s + 1) * PER]
            canvas = np.full((ROWS * sz, COLS * sz, 3), 255, np.uint8)
            for k, it in enumerate(chunk):
                gidx = s * PER + k
                tile = it["crop"].copy()
                cv2.putText(tile, str(gidx), (2, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
                canvas[(k // COLS) * sz:(k // COLS) * sz + sz,
                       (k % COLS) * sz:(k % COLS) * sz + sz] = tile
                manifest.append({"idx": gidx, "file": it["file"],
                                 "line": it["line"],
                                 "susp": round(it["exc"] + it.get("adev", 0), 3),
                                 "exc": it["exc"], "src": it["src"]})
            cv2.imencode(".jpg", canvas)[1].tofile(
                os.path.join(d, f"sheet_{s + 1:03d}.jpg"))
        json.dump(manifest, open(
            os.path.join(review_dir, f"manifest_{tag}.json"), "w"), indent=1)
        if items:
            # Cells breaking the count rule are where the mistakes live; tell
            # the reviewer exactly how far they must go before it thins out.
            nexc = sum(1 for it in items if it["exc"] > 0)
            hot = (nexc + PER - 1) // PER
            extra = (f"  <- vi pham luat dem: {nexc} o = sheet 001-{hot:03d}"
                     if sort_susp and nexc else "")
            print(f"  {tag:8s}: {len(items)} crop, {nsheets} sheet{extra}")
    rel = os.path.relpath(review_dir, ROOT).replace("\\", "/")
    print(f"\n-> Review sheets: {rel}/<N.tag>/sheet_NNN.jpg")
    print(f"   Fix:  python scripts/apply_review.py <tag> '<idx>=<class> ...' "
          f"--dir {rel}")
    print(f"   Then: python scripts/weekly_ingest.py --batch {os.path.relpath(WEEKLY, ROOT)} --merge")


def merge():
    """Purge sentinel-99 lines, then move incoming/* into train/."""
    si = os.path.join(STAGE, "images")
    sl = os.path.join(STAGE, "labels")
    if not os.path.isdir(si):
        print("Nothing staged (no incoming/). Run ingest first.")
        return
    ti = os.path.join(DATASET, "train", "images")
    tl = os.path.join(DATASET, "train", "labels")
    os.makedirs(ti, exist_ok=True)
    os.makedirs(tl, exist_ok=True)

    # purge sentinel 99 in staged labels first
    purged = 0
    for lp in glob.glob(os.path.join(sl, "*.txt")):
        lines = open(lp, encoding="utf-8").read().splitlines()
        keep = [l for l in lines if not (l.split() and l.split()[0] == "99")]
        if len(keep) != len(lines):
            purged += len(lines) - len(keep)
            open(lp, "w", encoding="utf-8").write(
                "\n".join(keep) + ("\n" if keep else ""))

    n = 0
    for ip in glob.glob(os.path.join(si, "*")):
        base = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(sl, base + ".txt")
        shutil.move(ip, os.path.join(ti, os.path.basename(ip)))
        if os.path.exists(lp):
            shutil.move(lp, os.path.join(tl, base + ".txt"))
        n += 1
    # clean up empty staging dirs
    for d in (si, sl, STAGE):
        try:
            os.rmdir(d)
        except OSError:
            pass
    print(f"Merged {n} images into train/ (purged {purged} sentinel boxes).")
    print("Now retrain: python scripts/train_items.py --data "
          "data/items_v20/data.yaml --name items_vNEXT --img-size 960 "
          "--batch-size 12 --workers 2 --no-deploy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None,
                    help="period workspace dir (e.g. ingest/2026-07-11). Scopes "
                         "inbox/staging/review under it. Default: weekly/ (legacy)")
    ap.add_argument("--period", default=None,
                    help="staged-name tag for a flat inbox (default: --batch "
                         "basename). Groups this period's images in the trainset.")
    ap.add_argument("--input", default=None,
                    help="folder to ingest (default: <batch>/inbox/). Recursive.")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--model", default=None,
                    help="detector weights (default: settings.ITEMS_MODEL). "
                         "Pass a .onnx to run the deployed ONNX model.")
    ap.add_argument("--device", default=None,
                    help="cpu | 0 | ... (default: auto; .onnx forced to cpu "
                         "unless onnxruntime-gpu is installed)")
    ap.add_argument("--merge", action="store_true",
                    help="finalize: purge sentinels + move staging into train/")
    ap.add_argument("--gallery-only", action="store_true",
                    help="skip detect: rebuild review sheets from the CURRENT "
                         "staging labels (use after a review pass — fixed boxes "
                         "move to their new class, sentinels drop out)")
    ap.add_argument("--review-dir", default=None,
                    help="where to write the sheets (default <batch>/review). "
                         "Use a dated dir for a fresh pass, e.g. "
                         "ingest/2026-07-11/review_2026-07-30")
    ap.add_argument("--tile", type=int, default=None,
                    help=f"tile size in px (default {SZ})")
    ap.add_argument("--skip-reviewed", default=None,
                    help="path to a reviewed.json progress file; every "
                         "cell already reviewed in the gallery it names is "
                         "left out of the rebuild (see load_reviewed)")
    ap.add_argument("--sort-susp", action="store_true",
                    help="order each class worst-first (count-rule violations, "
                         "then odd-sized crops, then generals/advisors outside "
                         "the palace) so the mistakes land on the first sheets")
    args = ap.parse_args()

    # --batch scopes every path under one period folder (isolated per period).
    global WEEKLY, INBOX, STAGE, REVIEW_DIR
    period = args.period
    if args.batch:
        WEEKLY = os.path.abspath(args.batch)
        INBOX = os.path.join(WEEKLY, "inbox")
        STAGE = os.path.join(WEEKLY, "staging")
        REVIEW_DIR = os.path.join(WEEKLY, "review")
        if not period:
            period = os.path.basename(os.path.normpath(WEEKLY))
    input_dir = args.input or INBOX

    if args.merge:
        merge()
        return
    if args.gallery_only:
        skip = None
        if args.skip_reviewed:
            print("Reading review progress ...")
            skip = load_reviewed(args.skip_reviewed)
            print(f"  -> {len(skip)} o se KHONG xuat hien lai")
            print()
        print("Rebuilding review galleries from current staging labels ...")
        build_galleries(args.review_dir and os.path.abspath(args.review_dir),
                        args.tile, args.sort_susp, skip)
        return
    week = detect_and_stage(input_dir, args.conf, args.model, args.device, period)
    if week:
        print("\nBuilding review galleries ...")
        build_galleries(sort_susp=args.sort_susp)


if __name__ == "__main__":
    main()
