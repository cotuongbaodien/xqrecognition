"""Split the dataset into OK (clean) vs NOT-OK (needs label fix) folders.

A source image is NOT-OK if it appears in the missing-label list (>= --missing-conf)
or the wrong-class list (>= --wrong-conf). Everything else is OK. All augmented
copies of a source follow their source so train/val integrity is kept.

Train only on the OK folder (less label noise); fix the NOT-OK folder on Roboflow.

    python scripts/split_clean.py --wrong-conf 0.95          # missing + only top wrong
    python scripts/split_clean.py --wrong-conf 2.0           # missing only (ignore wrong)
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}


def roboflow_name(fn):
    """Dataset filename -> bare source name (matches CSV roboflow_name)."""
    base = re.split(r"\.rf\.", fn)[0]
    parts = base.split("_")
    while parts and parts[-1].lower() in _EXTS:
        parts.pop()
    return "_".join(parts)


def flagged_from(csv_path, conf_min):
    out = set()
    if not csv_path.exists() or conf_min > 1.0:
        return out
    for r in csv.DictReader(csv_path.open(encoding="utf-8-sig")):
        if float(r["conf"]) >= conf_min:
            out.add(r["roboflow_name"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/items_v12")
    ap.add_argument("--missing", default="indep_missing_labels.csv")
    ap.add_argument("--wrong", default="indep_wrong_class.csv")
    ap.add_argument("--missing-conf", type=float, default=0.0)
    ap.add_argument("--wrong-conf", type=float, default=0.95)
    ap.add_argument("--out-ok", default="data/items_clean_ok")
    ap.add_argument("--out-notok", default="data/items_clean_notok")
    ap.add_argument("--list-only", action="store_true",
                    help="just write ok/notok source lists, don't copy files")
    args = ap.parse_args()

    flagged = (flagged_from(PROJECT_ROOT / args.missing, args.missing_conf)
               | flagged_from(PROJECT_ROOT / args.wrong, args.wrong_conf))
    print(f"Flagged sources (NOT-OK): {len(flagged)}")

    # gather all (image, label) across splits, grouped by source
    data = PROJECT_ROOT / args.data
    all_imgs = []
    for split in ("train", "valid", "test"):
        idir = data / split / "images"
        ldir = data / split / "labels"
        if idir.exists():
            all_imgs += [(p, ldir / (p.stem + ".txt")) for p in sorted(idir.glob("*"))
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    ok_src, notok_src = set(), set()
    ok_files, notok_files = [], []
    for img, lbl in all_imgs:
        name = roboflow_name(img.name)
        if name in flagged:
            notok_files.append((img, lbl)); notok_src.add(name)
        else:
            ok_files.append((img, lbl)); ok_src.add(name)

    print(f"OK    : {len(ok_src)} sources, {len(ok_files)} files")
    print(f"NOT-OK: {len(notok_src)} sources, {len(notok_files)} files")

    if args.list_only:
        (PROJECT_ROOT / "split_ok_sources.txt").write_text("\n".join(sorted(ok_src)), encoding="utf-8")
        (PROJECT_ROOT / "split_notok_sources.txt").write_text("\n".join(sorted(notok_src)), encoding="utf-8")
        print("Wrote split_ok_sources.txt / split_notok_sources.txt")
        return

    for out, files in ((args.out_ok, ok_files), (args.out_notok, notok_files)):
        base = PROJECT_ROOT / out
        if base.exists():
            shutil.rmtree(base)
        (base / "images").mkdir(parents=True)
        (base / "labels").mkdir(parents=True)
        for img, lbl in files:
            shutil.copy2(img, base / "images" / img.name)
            if lbl.exists():
                shutil.copy2(lbl, base / "labels" / lbl.name)
        print(f"  -> {out}/ ({len(files)} imgs)")


if __name__ == "__main__":
    main()
