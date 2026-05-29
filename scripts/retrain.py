"""End-to-end retrain pipeline.

When a new dataset zip arrives from Roboflow, run:

    python scripts/retrain.py --zip data/itemdetection260520.yolov8.zip

Does: extract → split → backup current model → train → deploy → test → log.
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to Roboflow zip")
    parser.add_argument("--name", default=None,
                        help="Run name (default: items_v{auto from date})")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true",
                        help="Only extract+split, skip training")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        sys.exit(f"Zip not found: {zip_path}")

    # Auto-name: items_vYYMMDD
    name = args.name or f"items_v{datetime.now().strftime('%y%m%d')}"
    dest = PROJECT_ROOT / "data" / name

    print(f"=== Retrain pipeline: {name} ===")
    print(f"Zip:  {zip_path}")
    print(f"Dest: {dest}")

    # 1. Extract
    if dest.exists():
        print(f"Removing existing {dest}")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    print(f"\n[1/5] Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    n_imgs = len(list((dest / "train" / "images").glob("*")))
    print(f"  → {n_imgs} images extracted")

    # 2. Split 80/15/5
    print(f"\n[2/5] Splitting 80/15/5...")
    subprocess.check_call([
        sys.executable, "scripts/split_items.py", "--dir", f"data/{name}"
    ], cwd=PROJECT_ROOT)

    if args.skip_train:
        print("\nSkip-train flag set. Done.")
        return

    # 3. Backup current model
    print(f"\n[3/5] Backing up current model...")
    current_model = PROJECT_ROOT / "boarddetection" / "models" / "items.pt"
    backup_dir = PROJECT_ROOT / "models" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if current_model.exists():
        backup_path = backup_dir / f"items_pre_{name}.pt"
        shutil.copy(current_model, backup_path)
        print(f"  → {backup_path}")
    else:
        print("  No existing model to backup.")

    # 4. Train
    print(f"\n[4/5] Training {args.epochs} epochs on {args.device}...")
    subprocess.check_call([
        sys.executable, "scripts/train_items.py",
        "--data", f"data/{name}/data.yaml",
        "--pretrained", "yolo11s.pt",
        "--epochs", str(args.epochs),
        "--batch-size", "16",
        "--device", args.device,
        "--seed", "42",
        "--name", name,
    ], cwd=PROJECT_ROOT)

    # 5. Test — output into a version-named folder so each build keeps its
    # own visualizations for side-by-side comparison (test/output_<name>/).
    out_dir = f"test/output_{name}"
    print(f"\n[5/5] Testing on test/ images → {out_dir}/ ...")
    test_dir = PROJECT_ROOT / "test"
    if test_dir.exists() and any(test_dir.glob("*.png")) or any(test_dir.glob("*.jpg")):
        subprocess.check_call([
            sys.executable, "detect.py",
            "--dir", "test",
            "--output", out_dir,
            "--confidence", "0.3",
        ], cwd=PROJECT_ROOT)
    else:
        print("  No test/ images found. Skipping.")

    print(f"\n=== Done. Model: boarddetection/models/items.pt ===")
    print(f"Visualizations: {out_dir}/")
    print(f"Backup: models/backups/items_pre_{name}.pt")
    print(f"\nIf results worse than baseline, rollback:")
    print(f"  cp models/backups/items_pre_{name}.pt boarddetection/models/items.pt")


if __name__ == "__main__":
    main()
