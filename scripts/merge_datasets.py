"""Merge multiple v6-format datasets into one output directory.

Usage:
    python scripts/merge_datasets.py --out data/items_v7 \\
        --src data/items_v6 --src data/pseudo_vn \\
        --src data/pseudo_chess --src data/pseudo_xiangqi
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", action="append", required=True,
                        help="Source dataset dir (must have train/images, train/labels)")
    parser.add_argument("--out", required=True,
                        help="Output dataset dir (will be created)")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "train" / "images").mkdir(parents=True)
    (out / "train" / "labels").mkdir(parents=True)

    total = 0
    for src in args.src:
        src = Path(src)
        prefix = src.name
        n = 0
        for split in ("train", "valid", "test"):
            img_dir = src / split / "images"
            lbl_dir = src / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                stem = img_path.stem
                new_name = f"{prefix}_{stem}"
                shutil.copy(img_path,
                            out / "train" / "images" / f"{new_name}{img_path.suffix}")
                lbl_path = lbl_dir / f"{stem}.txt"
                if lbl_path.exists():
                    shutil.copy(lbl_path,
                                out / "train" / "labels" / f"{new_name}.txt")
                n += 1
        print(f"  {src.name}: {n} files")
        total += n

    print(f"\nTotal merged: {total} images → {out}")


if __name__ == "__main__":
    main()
