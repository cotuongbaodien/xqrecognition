"""Second-opinion pass over staged pseudo-labels -> which cells are worth a look.

The pseudo-labels come from ONE detector run (imgsz=960, no TTA, conf 0.25).
Running the same weights again under a different view (bigger imgsz + flip/scale
TTA) gives a nearly-free second vote per box. Where the two votes agree with a
strong confidence the cell is almost never wrong, so a review gallery built from
the DISAGREEMENTS only is a small fraction of the full one.

  # 1. predict (writes <batch>/second_opinion.json)
  python scripts/second_opinion.py --batch ingest/2026-07-11
  # 2. how good is the signal? measure it against cells a human already fixed:
  python scripts/second_opinion.py --batch ingest/2026-07-11 --eval \
      --manifest ingest/_archive/2026-07-11/review_2026-08-17/manifest_tuongden.json \
      --orig-class 3 --max-idx 3899
  # 3. cells the two votes agree on -> feed to weekly_ingest as "don't show me"
  python scripts/second_opinion.py --batch ingest/2026-07-11 --write-agree agree.json

A box is FLAGGED (worth human eyes) when the second run
  * predicts a different class, or
  * finds nothing there at all (no box with IoU >= --iou), or
  * agrees but with confidence below --conf-ok.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boarddetection.settings import ITEM_CLASSES, ITEMS_MODEL  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ar if ar > 0 else 0.0


def predict(stage, out_path, only_srcs=None, imgsz=1280, tta=True, conf=0.10,
            model_path=None, device=None):
    """Run the second view and record, per staged box, what it votes for."""
    from ultralytics import YOLO
    imgs = sorted(p for p in glob.glob(os.path.join(stage, "images", "*"))
                  if p.lower().endswith(IMG_EXT))
    if only_srcs is not None:
        imgs = [p for p in imgs if os.path.basename(p) in only_srcs]
    mpath = model_path or str(ITEMS_MODEL)
    print(f"Loading {mpath} ...")
    model = YOLO(mpath)
    print(f"Second opinion on {len(imgs)} images "
          f"(imgsz={imgsz}, tta={tta}, conf>={conf}) ...")

    out, n_box = {}, 0
    for k, ip in enumerate(imgs):
        if k and k % 200 == 0:
            print(f"  {k}/{len(imgs)}")
        lp = os.path.join(stage, "labels",
                          os.path.splitext(os.path.basename(ip))[0] + ".txt")
        if not os.path.exists(lp):
            continue
        im = imread_u(ip)
        if im is None:
            continue
        H, W = im.shape[:2]
        res = model(im, conf=conf, imgsz=imgsz, augment=tta,
                    verbose=False, device=device)[0]
        pred = []
        if res.boxes is not None:
            for i in range(len(res.boxes)):
                cid = int(res.boxes.cls[i].cpu().numpy())
                if cid not in ITEM_CLASSES:
                    continue
                pred.append((cid, float(res.boxes.conf[i].cpu().numpy()),
                             res.boxes.xyxy[i].cpu().numpy().tolist()))
        rel = os.path.relpath(lp, ROOT).replace("\\", "/")
        rows = {}
        for li, ln in enumerate(open(lp, encoding="utf-8").read().splitlines()):
            p = ln.split()
            if not p or int(p[0]) not in ITEM_CLASSES:
                continue
            x, y, w, h = [float(v) for v in p[1:5]]
            box = [(x - w / 2) * W, (y - h / 2) * H,
                   (x + w / 2) * W, (y + h / 2) * H]
            best, biou = None, 0.0
            for cid, cf, pb in pred:
                v = _iou(box, pb)
                if v > biou:
                    best, biou = (cid, cf), v
            rows[li] = [best[0] if best else -1,
                        round(best[1], 4) if best else 0.0, round(biou, 3)]
            n_box += 1
        out[rel] = rows
    json.dump({"imgsz": imgsz, "tta": tta, "conf": conf, "boxes": out},
              open(out_path, "w"), indent=0)
    print(f"-> {n_box} box, ghi {os.path.relpath(out_path, ROOT)}")
    return out


def flagged(vote, cur_cls, iou_min, conf_ok):
    """True = the two views do not confidently agree -> show it to the human."""
    pcid, pconf, piou = vote
    if pcid < 0 or piou < iou_min:
        return True, "khong thay box"
    if pcid != cur_cls:
        return True, "khac lop"
    if pconf < conf_ok:
        return True, "conf thap"
    return False, ""


def rank_score(vote, cur_cls, iou_min):
    """How safe does the second view think this cell is? 0 = worst.

    Ordering a gallery by this puts the cells the two views argue about on the
    first sheets, then the ones they agree on but weakly, so a reviewer who
    stops early has still seen the worst of it.
    """
    pcid, pconf, piou = vote
    if pcid < 0 or piou < iou_min:
        return 0.02                      # nothing there on the second look
    if pcid != cur_cls:
        return 0.0                       # outright disagreement
    return pconf


def evaluate(votes, manifest, orig_class, max_idx, iou_min, conf_ok):
    """Score the signal against cells a human already went through.

    Every cell in `manifest` up to `max_idx` was pseudo-labelled `orig_class`
    and then eyeballed; the ones whose class differs today are the mistakes the
    reviewer found. Recall = share of those the flag would have shown them.
    """
    man = [e for e in json.load(open(manifest, encoding="utf-8"))
           if e["idx"] <= max_idx]
    cache = {}
    n = nerr = nflag = nerr_flag = 0
    missed, reasons = [], {}
    for e in man:
        f, li = e["file"], e["line"]
        if f not in cache:
            path = os.path.join(ROOT, f)
            cache[f] = (open(path, encoding="utf-8").read().splitlines()
                        if os.path.exists(path) else [])
        L = cache[f]
        if li >= len(L) or not L[li].split():
            continue
        now = int(L[li].split()[0])
        vote = votes.get(f, {}).get(str(li)) or votes.get(f, {}).get(li)
        if vote is None:
            continue
        n += 1
        # the human's verdict: changed class (or 99 'bo') = pseudo-label was wrong
        is_err = now != orig_class
        fl, why = flagged(vote, orig_class, iou_min, conf_ok)
        nerr += is_err
        nflag += fl
        if is_err and fl:
            nerr_flag += 1
            reasons[why] = reasons.get(why, 0) + 1
        elif is_err:
            missed.append((e["idx"], now))
    print(f"\n== Do tren {n} o da co nguoi soi ==")
    print(f"loi that (nguoi sua):      {nerr}  ({nerr / n:.2%})")
    print(f"bi flag:                   {nflag}  ({nflag / n:.2%} khoi luong con lai)")
    print(f"loi that ma flag bat duoc: {nerr_flag}/{nerr} = RECALL {nerr_flag / max(nerr, 1):.1%}")
    print(f"do sach cua flag:          {nerr_flag}/{max(nflag, 1)} = {nerr_flag / max(nflag, 1):.1%} flag la loi that")
    print(f"ly do bat duoc: {reasons}")
    if missed:
        print(f"LOT LUOI ({len(missed)}): idx dau -> lop dung: {missed[:25]}")
    return n, nerr, nflag, nerr_flag


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="ingest/<period> folder")
    ap.add_argument("--out", default=None, help="default <batch>/second_opinion.json")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iou", type=float, default=0.35,
                    help="min IoU to call it the same box")
    ap.add_argument("--conf-ok", type=float, default=0.85,
                    help="agreement below this confidence is still flagged")
    ap.add_argument("--conf-ok-also", type=float, default=0.60,
                    help="same, for the --also vote file (the weaker model "
                         "fires too often at a high threshold)")
    ap.add_argument("--repredict", action="store_true",
                    help="re-run the model even if the vote json already exists")
    ap.add_argument("--eval", action="store_true",
                    help="score the flag against an already-reviewed manifest")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--orig-class", type=int, default=None)
    ap.add_argument("--max-idx", type=int, default=10 ** 9)
    ap.add_argument("--write-rank", default=None,
                    help="write {'file|line': score} used to sort a gallery "
                         "worst-first (feed to weekly_ingest --rank-pairs). "
                         "Combine several vote files with --also")
    ap.add_argument("--also", default=None,
                    help="a second vote json (e.g. the other model's run); "
                         "agree/rank then take the WORST of the two votes")
    ap.add_argument("--write-agree", default=None,
                    help="write [[file, line], ...] of confidently-agreed cells "
                         "(feed to weekly_ingest --skip-pairs)")
    args = ap.parse_args()

    batch = os.path.abspath(args.batch)
    stage = os.path.join(batch, "staging")
    out_path = args.out or os.path.join(batch, "second_opinion.json")

    only = None
    if args.eval and args.manifest:
        man = json.load(open(args.manifest, encoding="utf-8"))
        only = {e["src"] for e in man if e["idx"] <= args.max_idx}
        print(f"Calibration subset: {len(only)} anh")

    if os.path.exists(out_path) and not args.repredict:
        print(f"Dung lai {os.path.relpath(out_path, ROOT)} "
              f"(--repredict de chay lai model)")
        votes = json.load(open(out_path))["boxes"]
    else:
        votes = predict(stage, out_path, only, args.imgsz, not args.no_tta,
                        args.conf, args.model, args.device)

    if args.eval:
        evaluate(votes, args.manifest, args.orig_class, args.max_idx,
                 args.iou, args.conf_ok)

    if args.write_agree or args.write_rank:
        other = json.load(open(args.also))["boxes"] if args.also else {}
        cache, agree, rank = {}, [], {}
        n = 0
        for f, rows in votes.items():
            path = os.path.join(ROOT, f)
            if f not in cache:
                cache[f] = (open(path, encoding="utf-8").read().splitlines()
                            if os.path.exists(path) else [])
            L = cache[f]
            for li, vote in rows.items():
                li = int(li)
                if li >= len(L) or not L[li].split():
                    continue
                cur = int(L[li].split()[0])
                votes2 = [(vote, args.conf_ok)]
                v2 = other.get(f, {}).get(str(li))
                if v2 is not None:
                    votes2.append((v2, args.conf_ok_also))
                n += 1
                if not any(flagged(v, cur, args.iou, ck)[0] for v, ck in votes2):
                    agree.append([f, li])
                rank[f"{f}|{li}"] = round(
                    min(rank_score(v, cur, args.iou) for v, _ in votes2), 4)
        if args.write_agree:
            p = os.path.join(batch, args.write_agree)
            json.dump(agree, open(p, "w"), indent=0)
            print(f"-> {len(agree)}/{n} o hai ben dong y ({len(agree) / max(n, 1):.1%}), "
                  f"ghi {os.path.relpath(p, ROOT)}")
        if args.write_rank:
            p = os.path.join(batch, args.write_rank)
            json.dump(rank, open(p, "w"), indent=0)
            print(f"-> diem xep hang {len(rank)} o, ghi {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
