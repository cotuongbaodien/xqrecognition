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

# Self-contained workspace for the whole weekly flow (gitignored).
#   weekly/inbox/2026-WNN/images/  <- COPY server images here
#   weekly/staging/{images,labels} <- pseudo-labeled, pre-review (auto)
#   weekly/review/<N.tag>/sheet_*  <- label review galleries (auto)
#   weekly/test_candidates/        <- test picks (pick_test_candidates.py)
# On --merge, staging/ moves into the canonical trainset data/items_v20/train/.
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


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _week_of(path, input_dir):
    """Derive the week tag from an image path. Handles both a flat folder and a
    parent dir of week-subfolders (e.g. ocrservice/2026-W24/images/x.jpg)."""
    rel = os.path.relpath(path, input_dir).replace("\\", "/").split("/")
    if len(rel) > 1:
        return rel[0]                                   # week subfolder name
    return os.path.basename(os.path.normpath(input_dir))


def detect_and_stage(input_dir, conf):
    from ultralytics import YOLO
    # recursive: --input can be ONE week folder OR a parent of week-subfolders
    # (each with images/). Skip anything under a labels/ dir.
    imgs = [p for p in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
            if p.lower().endswith(IMG_EXT)
            and (os.sep + "labels" + os.sep) not in p]
    if not imgs:
        print(f"No images in {input_dir}")
        return None
    weeks = sorted({_week_of(p, input_dir) for p in imgs})
    print(f"Weeks: {', '.join(weeks)}")
    os.makedirs(os.path.join(STAGE, "images"), exist_ok=True)
    os.makedirs(os.path.join(STAGE, "labels"), exist_ok=True)

    print(f"Loading {ITEMS_MODEL} ...")
    model = YOLO(str(ITEMS_MODEL))
    print(f"Detecting {len(imgs)} images (conf>={conf}, imgsz=960) ...")

    dist = Counter()
    n_imgs = n_box = 0
    for ip in imgs:
        im = imread_u(ip)
        if im is None:
            print(f"  skip unreadable: {ip}")
            continue
        H, W = im.shape[:2]
        res = model(im, conf=conf, imgsz=960, verbose=False)[0]
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
        base = f"weekly_{_week_of(ip, input_dir)}_{stem}"
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


def build_galleries():
    """Per-class review sheets + manifests for the staged incoming labels.
    Mirrors review_gallery.py output so apply_review.py works unchanged."""
    import json
    per_class = defaultdict(list)   # cid -> [(crop, relfile, line, src)]
    for ip in glob.glob(os.path.join(STAGE, "images", "*")):
        lp = os.path.join(STAGE, "labels",
                          os.path.splitext(os.path.basename(ip))[0] + ".txt")
        if not os.path.exists(lp):
            continue
        im = imread_u(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        for li, ln in enumerate(open(lp, encoding="utf-8")):
            p = ln.split()
            if not p:
                continue
            cid = int(p[0])
            if cid not in CID2TAG:
                continue
            x, yy, w, h = [float(v) for v in p[1:5]]
            x1, y1 = int((x - w / 2) * W), int((yy - h / 2) * H)
            x2, y2 = int((x + w / 2) * W), int((yy + h / 2) * H)
            cr = im[max(0, y1):y2, max(0, x1):x2]
            if cr.size == 0:
                continue
            rel = os.path.relpath(lp, ROOT).replace("\\", "/")
            per_class[cid].append((cv2.resize(cr, (SZ, SZ)), rel, li,
                                   os.path.basename(ip)))

    os.makedirs(REVIEW_DIR, exist_ok=True)
    for cid, tag in CID2TAG.items():
        items = per_class.get(cid, [])
        d = os.path.join(REVIEW_DIR, f"{PREFIX[tag]}.{tag}")
        os.makedirs(d, exist_ok=True)
        for old in glob.glob(os.path.join(d, "sheet_*.jpg")):
            os.remove(old)
        manifest = []
        nsheets = (len(items) + PER - 1) // PER
        for s in range(nsheets):
            chunk = items[s * PER:(s + 1) * PER]
            canvas = np.full((ROWS * SZ, COLS * SZ, 3), 255, np.uint8)
            for k, (cr, f, li, src) in enumerate(chunk):
                gidx = s * PER + k
                tile = cr.copy()
                cv2.putText(tile, str(gidx), (2, 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                canvas[(k // COLS) * SZ:(k // COLS) * SZ + SZ,
                       (k % COLS) * SZ:(k % COLS) * SZ + SZ] = tile
                manifest.append({"idx": gidx, "file": f, "line": li,
                                 "susp": 0.0, "src": src})
            cv2.imencode(".jpg", canvas)[1].tofile(
                os.path.join(d, f"sheet_{s + 1:03d}.jpg"))
        json.dump(manifest, open(
            os.path.join(REVIEW_DIR, f"manifest_{tag}.json"), "w"), indent=1)
        if items:
            print(f"  {tag:8s}: {len(items)} crop, {nsheets} sheet")
    print(f"\n-> Review sheets: weekly/review/<N.tag>/sheet_NNN.jpg")
    print("   Fix:  python scripts/apply_review.py <tag> '<idx>=<class> ...' "
          "--dir weekly/review")
    print("   Test: python scripts/pick_test_candidates.py --n 20")
    print("   Then: python scripts/weekly_ingest.py --merge")


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
    ap.add_argument("--input", default=INBOX,
                    help="folder to ingest (default: weekly/inbox/, with "
                         "2026-WNN/images subfolders). Recursive.")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--merge", action="store_true",
                    help="finalize: purge sentinels + move staging into train/")
    args = ap.parse_args()

    if args.merge:
        merge()
        return
    week = detect_and_stage(args.input, args.conf)
    if week:
        print("\nBuilding review galleries ...")
        build_galleries()


if __name__ == "__main__":
    main()
