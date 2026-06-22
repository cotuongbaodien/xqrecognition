"""Consolidate ALL piece crops into ONE clean, standardized library.

Reads every digital skin (PNG_hires + rasterized svg_png, many with Chinese
folder names) + the harvested real crops, and writes them ALL as uniform
RGBA PNGs with ASCII filenames under data/piece_crops_all/<class_name>/.

This removes the per-run pain of mixed formats + non-ASCII paths (cv2 on
Windows fails on Chinese paths). After this, synth_gen reads only this one
ASCII folder. Run once; rebuild only when assets change.

Usage: python scripts/consolidate_crops.py
"""

import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from boarddetection.settings import ITEM_CLASSES  # noqa: E402

FEN2NAME = {fen: ITEM_CLASSES[c][0] for c, (_, fen) in ITEM_CLASSES.items() if fen}
# (color-letter, piece-letter) -> class name
DIGI2NAME = {}
for c, (name, fen) in ITEM_CLASSES.items():
    if fen:
        DIGI2NAME[("b" if fen.islower() else "r", fen.lower())] = name

DIGITAL_ROOTS = ["download/BO_COTUONG_CLEAN/PNG_hires", "data/piece_svg_png",
                 "download/BO_COTUONG_CLEAN/_lowres_54px"]
REAL_ROOT = "data/piece_crops_real"
OUT = "data/piece_crops_all"


def imread_u(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), flags)
    except Exception:
        return None


def imwrite_u(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok


def main():
    out = PROJECT_ROOT / OUT
    if out.exists():
        import shutil
        shutil.rmtree(out)
    for name in set(FEN2NAME.values()):
        (out / name).mkdir(parents=True, exist_ok=True)

    skin_map = {}   # skin_idx -> original skin folder name (for reference)
    counts = {n: 0 for n in set(FEN2NAME.values())}

    # digital skins (ASCII-index the skin to keep output filenames ASCII)
    skin_idx = 0
    for root in DIGITAL_ROOTS:
        for skin_dir in sorted(glob.glob(f"{PROJECT_ROOT / root}/*")):
            if not os.path.isdir(skin_dir):
                continue
            skin = os.path.basename(skin_dir)
            tag = f"d{skin_idx:03d}"
            skin_map[tag] = skin
            got = False
            for f in glob.glob(f"{skin_dir}/*.png"):
                base = os.path.splitext(os.path.basename(f))[0].lower()
                if len(base) != 2 or base[0] not in "br" or base[1] not in "kabnrcp":
                    continue
                name = DIGI2NAME.get((base[0], base[1]))
                if name is None:
                    continue
                im = imread_u(f)
                if im is None or im.ndim != 3 or im.shape[2] != 4:
                    continue
                imwrite_u(str(out / name / f"{tag}_{base}.png"), im)
                counts[name] += 1
                got = True
            if got:
                skin_idx += 1

    # real harvested crops (already ASCII; just copy through, re-tag)
    for name in set(FEN2NAME.values()):
        for f in sorted(glob.glob(f"{PROJECT_ROOT / REAL_ROOT}/{name}/*.png")):
            im = imread_u(f)
            if im is None or im.ndim != 3 or im.shape[2] != 4:
                continue
            stem = os.path.splitext(os.path.basename(f))[0]
            imwrite_u(str(out / name / f"real_{stem}.png"), im)
            counts[name] += 1

    (out / "skin_map.json").write_text(
        json.dumps(skin_map, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(counts.values())
    print(f"Consolidated {total} crops across {len(counts)} classes "
          f"({skin_idx} digital skins + real):")
    for n in sorted(counts):
        print(f"  {n:16s} {counts[n]}")
    print(f"Output: {out} (ASCII paths, uniform RGBA PNG)")
    print(f"skin_map.json maps d### tag -> original (Chinese) skin name")


if __name__ == "__main__":
    main()
