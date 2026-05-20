"""Split data/items_v{N}/train/ into train/valid/test 80/15/5 with seed=42."""

import argparse
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="data/items_v4")
args = parser.parse_args()

ROOT = Path(__file__).parent.parent / args.dir

CLASSES = [
    # v6+ alphabetical order (matches Roboflow export)
    "black-advisor", "black-cannon", "black-chariot", "black-elephant",
    "black-general", "black-horse", "black-soldier",
    "board-border", "board-conner",
    "palace-bottom", "palace-center", "palace-conner",
    "red-advisor", "red-cannon", "red-chariot", "red-elephant",
    "red-general", "red-horse", "red-soldier",
]

SEED = 42


def main():
    # Move original train/ → _src/ to free up the train/ name for split output
    src_dir = ROOT / "_src"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    orig_train = ROOT / "train"
    if not orig_train.exists():
        raise SystemExit(f"Expected {orig_train} from zip extract")
    orig_train.rename(src_dir)

    src_img = src_dir / "images"
    src_lbl = src_dir / "labels"

    images = sorted([p for p in src_img.glob("*")
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"Found {len(images)} images in source")

    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * 0.80)
    n_valid = int(n * 0.15)
    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_valid],
        "test":  images[n_train + n_valid:],
    }

    for split, files in splits.items():
        img_dir = ROOT / split / "images"
        lbl_dir = ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            lbl_path = src_lbl / (img_path.stem + ".txt")
            shutil.copy(img_path, img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_dir / lbl_path.name)
        print(f"  {split}: {len(files)} images")

    # Cleanup source
    shutil.rmtree(src_dir)

    # Write data.yaml
    yaml_text = (
        f"path: {ROOT.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
    )
    for i, name in enumerate(CLASSES):
        yaml_text += f"  {i}: {name}\n"
    (ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"\nWrote {ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
