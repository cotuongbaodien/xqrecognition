"""
Evaluate pipeline on fendata using mirror-tolerant metric.
A prediction is considered correct if either the original or its
horizontal mirror matches the ground truth FEN.
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import XiangqiRecognizer
from src.fen_generator import FENGenerator


def load_ground_truth(csv_path: Path) -> dict:
    gt = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fen = row["fen"].split(" ")[0] if " " in row["fen"] else row["fen"]
            if fen and len(fen) > 5:
                gt[row["image"]] = fen
    return gt


def evaluate(images_dir: Path, gt: dict, verbose: bool = True):
    rec = XiangqiRecognizer(use_board_detection=True)
    fg = FENGenerator()

    exact = 0
    mirror_match = 0
    fail = 0
    total = 0
    cell_accs = []

    failed_imgs = []

    for img_path in sorted(images_dir.glob("*.png")):
        if img_path.name not in gt:
            continue
        total += 1

        result = rec.recognize(str(img_path))
        pred = result.fen.split(" ")[0]
        gt_fen = gt[img_path.name]

        # Mirror version
        mirrored = fg.mirror_board_horizontal(result.board_state)
        pred_m = fg.generate_fen(mirrored).split(" ")[0]

        if pred == gt_fen:
            exact += 1
        elif pred_m == gt_fen:
            mirror_match += 1
        else:
            fail += 1
            comp_n = fg.compare_fen(pred + " w", gt_fen + " w")
            comp_m = fg.compare_fen(pred_m + " w", gt_fen + " w")
            best_acc = max(comp_n["accuracy"], comp_m["accuracy"])
            failed_imgs.append((img_path.name, best_acc, pred, pred_m, gt_fen))

        # Best of normal/mirror cell accuracy
        comp_n = fg.compare_fen(pred + " w", gt_fen + " w")
        comp_m = fg.compare_fen(pred_m + " w", gt_fen + " w")
        cell_accs.append(max(comp_n["accuracy"], comp_m["accuracy"]))

    avg_cell = sum(cell_accs) / len(cell_accs) if cell_accs else 0

    print("=" * 70)
    print(f"Evaluation on {total} images from {images_dir.name}/")
    print("=" * 70)
    print(f"Exact FEN match (no mirror needed):   {exact}/{total} ({exact/total*100:.1f}%)")
    print(f"Mirror match (after horizontal flip): {mirror_match}/{total} ({mirror_match/total*100:.1f}%)")
    print(f"Total OK (mirror-tolerant):           {exact + mirror_match}/{total} ({(exact + mirror_match)/total*100:.1f}%)")
    print(f"Failed:                               {fail}/{total} ({fail/total*100:.1f}%)")
    print(f"Avg cell accuracy:                    {avg_cell:.1%}")
    print(f">=95% cells:                          {sum(1 for a in cell_accs if a >= 0.95)}/{total}")
    print(f">=90% cells:                          {sum(1 for a in cell_accs if a >= 0.9)}/{total}")
    print(f">=80% cells:                          {sum(1 for a in cell_accs if a >= 0.8)}/{total}")

    if verbose and failed_imgs:
        print("\n" + "=" * 70)
        print(f"Failed images ({len(failed_imgs)}):")
        print("=" * 70)
        for name, acc, pred, pred_m, gt_fen in failed_imgs:
            print(f"\n{name} - best cell acc: {acc:.0%}")
            print(f"  pred normal: {pred}")
            print(f"  pred mirror: {pred_m}")
            print(f"  gt:          {gt_fen}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fendata", help="Path to data dir")
    parser.add_argument("--quiet", action="store_true", help="Don't show failed images")
    args = parser.parse_args()

    data_dir = Path(args.data)
    gt = load_ground_truth(data_dir / "labels.csv")
    images_dir = data_dir / "images"

    if not gt:
        print(f"No ground truth labels found in {data_dir}")
        return

    evaluate(images_dir, gt, verbose=not args.quiet)


if __name__ == "__main__":
    main()
