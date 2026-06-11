"""End-to-end retrain pipeline.

When a new dataset zip arrives from Roboflow, run:

    python scripts/retrain.py --zip data/itemdetection260520.yolov8.zip

Does: extract -> split -> backup current model -> train -> test -> FEN gate -> deploy.

The FEN gate exists because val mAP does NOT predict FEN accuracy (v15 had
higher mAP than v14 but scored 58/86 vs 61/86 exact-FEN). The new model is
deployed only if its mirror-tolerant exact-FEN score on test/ is at least
the current model's score.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fen_exact_score(results_json: Path, gt_path: Path):
    """Mirror-tolerant exact-FEN count, same logic as scripts/eval_fen.py."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    from eval_fen import diff, mirror_fen, parse_gt, stem_of

    gt = parse_gt(gt_path)
    results = json.load(open(results_json, encoding="utf-8"))
    det = {stem_of(r["image"]): r["fen"].split()[0]
           for r in results if "fen" in r}
    exact = n = 0
    for key, g in gt.items():
        if key not in det:
            continue
        n += 1
        d = det[key]
        if min(diff(d, g)[0], diff(d, mirror_fen(g))[0]) == 0:
            exact += 1
    return exact, n


def run_detect(out_dir: str, model_path: Path):
    subprocess.check_call([
        sys.executable, "detect.py",
        "--dir", "test",
        "--output", out_dir,
        "--confidence", "0.3",
        "--items-model", str(model_path),
    ], cwd=PROJECT_ROOT)
    return PROJECT_ROOT / out_dir / "results.json"


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
    print(f"\n[1/6] Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    n_imgs = len(list((dest / "train" / "images").glob("*")))
    print(f"  -> {n_imgs} images extracted")

    # 2. Split 80/15/5 -- but SKIP if the export is already pre-split by
    # Roboflow (has valid/ + test/). Re-splitting a pre-split set would
    # double-split it and corrupt the train/valid/test folders.
    pre_split = (dest / "valid" / "images").exists() and (dest / "test" / "images").exists()
    if pre_split:
        print(f"\n[2/6] Pre-split detected (train/valid/test present) -- skipping split_items.")
        classes = [
            "black-advisor", "black-cannon", "black-chariot", "black-elephant",
            "black-general", "black-horse", "black-soldier", "board-conner",
            "palace-bottom", "palace-center", "palace-conner", "red-advisor",
            "red-cannon", "red-chariot", "red-elephant", "red-general",
            "red-horse", "red-soldier",
        ]
        yaml_text = (
            f"path: {dest.resolve()}\n"
            "train: train/images\nval: valid/images\ntest: test/images\n\n"
            f"nc: {len(classes)}\nnames:\n"
        )
        for i, c in enumerate(classes):
            yaml_text += f"  {i}: {c}\n"
        (dest / "data.yaml").write_text(yaml_text, encoding="utf-8")
        for split in ("train", "valid", "test"):
            k = len(list((dest / split / "images").glob("*")))
            print(f"  {split}: {k} images")
    else:
        print(f"\n[2/6] Splitting 80/15/5...")
        subprocess.check_call([
            sys.executable, "scripts/split_items.py", "--dir", f"data/{name}"
        ], cwd=PROJECT_ROOT)

    if args.skip_train:
        print("\nSkip-train flag set. Done.")
        return

    # 3. Backup current model
    print(f"\n[3/6] Backing up current model...")
    current_model = PROJECT_ROOT / "boarddetection" / "models" / "items.pt"
    backup_dir = PROJECT_ROOT / "models" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if current_model.exists():
        backup_path = backup_dir / f"items_pre_{name}.pt"
        shutil.copy(current_model, backup_path)
        print(f"  -> {backup_path}")
    else:
        print("  No existing model to backup.")

    # 4. Train (no auto-deploy -- deploy is gated on FEN score below)
    print(f"\n[4/6] Training {args.epochs} epochs on {args.device}...")
    subprocess.check_call([
        sys.executable, "scripts/train_items.py",
        "--data", f"data/{name}/data.yaml",
        "--pretrained", "yolo11s.pt",
        "--epochs", str(args.epochs),
        "--batch-size", "16",
        "--device", args.device,
        "--seed", "42",
        "--name", name,
        "--no-deploy",
    ], cwd=PROJECT_ROOT)
    best = PROJECT_ROOT / "runs" / "items" / name / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"Training finished but best weights not found: {best}")

    # 5. Test the CANDIDATE model -- output into a version-named folder so
    # each build keeps its own visualizations (test/output_<name>/).
    out_dir = f"test/output_{name}"
    test_dir = PROJECT_ROOT / "test"
    gt_path = test_dir / "ground_truth.txt"
    has_images = test_dir.exists() and (
        any(test_dir.glob("*.png")) or any(test_dir.glob("*.jpg")))
    if not has_images or not gt_path.exists():
        # Can't gate without test set + ground truth: deploy unconditionally
        # (old behavior) but say so loudly.
        print("\n[5/6] No test images or ground_truth.txt -- FEN gate skipped, "
              "deploying unconditionally.")
        shutil.copy(best, current_model)
        print(f"\n=== Done. Deployed (ungated): {current_model} ===")
        return

    print(f"\n[5/6] Testing candidate on test/ -> {out_dir}/ ...")
    new_results = run_detect(out_dir, best)
    new_exact, n = fen_exact_score(new_results, gt_path)
    print(f"  Candidate {name}: {new_exact}/{n} exact-FEN (mirror-tolerant)")

    # 6. FEN gate -- compare against the currently deployed model. The
    # baseline score is cached by model hash so we only re-run detection
    # on the old model when it actually changed.
    print(f"\n[6/6] FEN gate vs current model...")
    cache_path = backup_dir / "fen_baseline.json"
    cur_hash = _md5(current_model) if current_model.exists() else None
    base = None
    if cur_hash and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("hash") == cur_hash:
            base = cached
    if base is None and cur_hash:
        print("  No cached baseline for current model -- evaluating it...")
        base_results = run_detect(f"test/output_{name}_baseline", current_model)
        b_exact, b_n = fen_exact_score(base_results, gt_path)
        base = {"hash": cur_hash, "exact": b_exact, "n": b_n}
        cache_path.write_text(json.dumps(base), encoding="utf-8")
    base_exact = base["exact"] if base else -1
    print(f"  Baseline (deployed model): {base_exact}/{base['n'] if base else 0}")

    if new_exact >= base_exact:
        shutil.copy(best, current_model)
        cache_path.write_text(
            json.dumps({"hash": _md5(current_model), "exact": new_exact, "n": n}),
            encoding="utf-8")
        print(f"\n=== DEPLOYED: {name} ({new_exact}/{n} >= {base_exact}) ===")
        print(f"Model: {current_model}")
    else:
        print(f"\n=== NOT DEPLOYED: {name} scored {new_exact}/{n} "
              f"< baseline {base_exact} ===")
        print(f"Current model kept. Candidate weights: {best}")
        print(f"Candidate backup: models/backups/items_{name}.pt")
    print(f"Visualizations: {out_dir}/")
    print(f"Pre-train backup: models/backups/items_pre_{name}.pt")


if __name__ == "__main__":
    main()

