#!/usr/bin/env python3
import shutil
import random
from pathlib import Path


def split_dataset(src_images, src_labels, dst, val_ratio=0.2, seed=42):
    random.seed(seed)
    imgs = sorted([p for p in Path(src_images).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    labels_dir = Path(src_labels)
    pairs = [(img, labels_dir / (img.stem + ".txt")) for img in imgs]
    random.shuffle(pairs)
    n_val = int(len(pairs)*val_ratio)
    train = pairs[n_val:]
    val = pairs[:n_val]
    for name, subset in [("train", train), ("valid", val)]:
        im_out = Path(dst)/name/"images"
        lb_out = Path(dst)/name/"labels"
        im_out.mkdir(parents=True, exist_ok=True)
        lb_out.mkdir(parents=True, exist_ok=True)
        for imgp, lbp in subset:
            shutil.copy(imgp, im_out/imgp.name)
            if lbp.exists():
                shutil.copy(lbp, lb_out/lbp.name)
            else:
                open(lb_out/(imgp.stem+".txt"), "w").close()


if __name__ == "__main__":
    split_dataset(
        "images",
        "labels",
        "dataset",
        val_ratio=0.15
    )
    print("done")
