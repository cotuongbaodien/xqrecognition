"""Pseudo-label landmarks on piece-only datasets via current model.

Given a Roboflow zip of full-board images with PIECE-only labels (in some
class-name convention), this script:

  1. Extracts the zip, reads its data.yaml class list
  2. Maps each input class to the v6 19-class scheme by fuzzy name match
     (case-insensitive, separator-agnostic — black-advisor == Advisor_black
     == advisor_black)
  3. Runs the current items.pt model on each image, takes landmark
     detections (board-conner, palace-bottom/center/conner, board-border)
     with confidence ≥ threshold as PSEUDO-LABELS
  4. Writes a new YOLO dataset (data/<name>/) that combines the user's
     piece labels (remapped to v6 IDs) with pseudo landmark labels
  5. Auto-splits 80/15/5 and writes data.yaml

Use:
    python scripts/pseudo_label_dataset.py \\
        --zip data/some_piece_only.yolov8.zip \\
        --name items_piecemixed \\
        --landmark-conf 0.5

Then retrain:
    python scripts/train_items.py --data data/items_piecemixed/data.yaml \\
        --name items_v7
"""

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from boarddetection.item_detector import ItemDetector  # noqa: E402
from boarddetection.settings import ITEM_CLASSES, PIECE_CLASS_IDS, LANDMARK_CLASS_IDS  # noqa: E402

V6_CLASSES = [ITEM_CLASSES[i][0] for i in range(len(ITEM_CLASSES))]


# Hardcoded mappings for known Xiangqi datasets with non-English class names.
# Add new entries when encountering new naming conventions.
EXPLICIT_MAP = {
    # `chess project.v13i` — Roboflow workspace numeric scheme
    # User confirmed: 1=general, 2=advisor, 3=elephant, 4=horse, 5=chariot,
    # 6=cannon, 7=soldier (Xiangqi traditional value order)
    "b_1": "black-general", "b_2": "black-advisor", "b_3": "black-elephant",
    "b_4": "black-horse",   "b_5": "black-chariot", "b_6": "black-cannon",
    "b_7": "black-soldier",
    "r_1": "red-general",   "r_2": "red-advisor",   "r_3": "red-elephant",
    "r_4": "red-horse",     "r_5": "red-chariot",   "r_6": "red-cannon",
    "r_7": "red-soldier",
    # `xiangqi.v9i` — Chinese pinyin
    "heijiang":  "black-general",  "heiju":    "black-chariot",
    "heima":     "black-horse",    "heipao":   "black-cannon",
    "heishi":    "black-advisor",  "heizu":    "black-soldier",
    "hongbing":  "red-soldier",    "hongju":   "red-chariot",
    "hongma":    "red-horse",      "hongpao":  "red-cannon",
    "hongshuai": "red-general",    "hongxiang":"red-elephant",
    "kuang":     None,  # drop
    # `XiangQi Chess.v11i` — Vietnamese abbrev (D=Đỏ=red, X=Xanh=black)
    "ma_d": "red-horse",    "ma_x": "black-horse",
    "ph_d": "red-cannon",   "ph_x": "black-cannon",
    "si_d": "red-advisor",  "si_x": "black-advisor",
    "tg_d": "red-general",  "tg_x": "black-general",
    "to_d": "red-soldier",  "to_x": "black-soldier",
    "tu_d": "red-elephant", "tu_x": "black-elephant",
    "xe_d": "red-chariot",  "xe_x": "black-chariot",
}


def normalize_classname(name: str) -> str:
    """Try explicit map first, then fuzzy match by English keywords.
    Returns canonical v6 class name like 'black-advisor', or None if dropped,
    or the raw normalized name if unmatched.
    """
    raw = name.lower().strip()
    # Explicit first
    if raw in EXPLICIT_MAP:
        return EXPLICIT_MAP[raw]

    s = re.sub(r"[_\s-]+", "-", raw)
    synonyms = {
        "knight": "horse", "rook": "chariot", "pawn": "soldier",
        "king": "general", "bishop": "elephant",
    }
    tokens = s.split("-")
    color = None
    piece = None
    for tok in tokens:
        if tok in ("red", "black"):
            color = tok
        else:
            tok_canon = synonyms.get(tok, tok)
            if tok_canon in ("advisor", "cannon", "chariot", "elephant",
                             "general", "horse", "soldier"):
                piece = tok_canon
    if color and piece:
        return f"{color}-{piece}"
    return s  # fallback (unmapped)


def build_class_map(src_classes):
    """Map src_class_id → v6_class_id (or None if dropped/unmapped)."""
    v6_lookup = {V6_CLASSES[i]: i for i in range(len(V6_CLASSES))}
    mapping = {}
    for src_id, src_name in enumerate(src_classes):
        canon = normalize_classname(src_name)
        if canon is None:
            mapping[src_id] = None
            print(f"  src {src_id:2} '{src_name}' → DROPPED (explicit)")
            continue
        v6_id = v6_lookup.get(canon)
        mapping[src_id] = v6_id
        status = (f"→ {V6_CLASSES[v6_id]} (id {v6_id})"
                  if v6_id is not None else f"UNMAPPED (canon={canon})")
        print(f"  src {src_id:2} '{src_name}' {status}")
    return mapping


def read_data_yaml_classes(yaml_path: Path):
    """Parse class names list from data.yaml — supports both formats:
       names: ['a', 'b']         (Roboflow inline list)
       names:                    (mapping format)
         0: a
         1: b
    """
    text = yaml_path.read_text(encoding="utf-8")
    # Try inline list format
    m = re.search(r"names\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if m:
        items = re.findall(r"'([^']+)'|\"([^\"]+)\"", m.group(1))
        return [a or b for a, b in items]
    # Try mapping format
    classes = []
    in_names = False
    for line in text.splitlines():
        if re.match(r"^names\s*:", line):
            in_names = True
            continue
        if in_names:
            m = re.match(r"^\s+\d+\s*:\s*(.+)$", line)
            if m:
                classes.append(m.group(1).strip().strip("'\""))
            else:
                if line and not line.startswith(" "):
                    break
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Roboflow YOLOv8 zip")
    parser.add_argument("--name", required=True, help="Output dataset name")
    parser.add_argument("--landmark-conf", type=float, default=0.5,
                        help="Min confidence for pseudo landmark labels")
    parser.add_argument("--model", default=None,
                        help="Path to model for pseudo-labeling (default: boarddetection/models/items.pt)")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        sys.exit(f"Not found: {zip_path}")

    # Extract zip
    src_dir = PROJECT_ROOT / "data" / f"_{args.name}_raw"
    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True)
    print(f"\n[1/4] Extracting {zip_path.name} → {src_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(src_dir)

    # Parse source class list
    yaml_path = src_dir / "data.yaml"
    if not yaml_path.exists():
        sys.exit(f"Missing data.yaml in extracted zip")
    src_classes = read_data_yaml_classes(yaml_path)
    print(f"\n[2/4] Source has {len(src_classes)} classes")
    class_map = build_class_map(src_classes)
    unmapped = [src_classes[i] for i, v in class_map.items() if v is None]
    if unmapped:
        print(f"\nWarning: {len(unmapped)} classes UNMAPPED — labels with these will be dropped:")
        for c in unmapped:
            print(f"  - {c}")

    # Load v6 model for pseudo-labeling
    model_path = args.model or str(PROJECT_ROOT / "boarddetection" / "models" / "items.pt")
    print(f"\n[3/4] Loading model for pseudo-labeling: {model_path}")
    det = ItemDetector(model_path)

    # Process all train images (Roboflow may have train/valid/test splits;
    # we merge everything into one pool and let split_items re-split later)
    out_dir = PROJECT_ROOT / "data" / args.name / "train"
    out_imgs = out_dir / "images"
    out_lbls = out_dir / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_landmarks_added = 0
    for split in ("train", "valid", "test"):
        split_imgs = src_dir / split / "images"
        split_lbls = src_dir / split / "labels"
        if not split_imgs.exists():
            continue
        for img_path in split_imgs.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            n_total += 1
            # Read original piece labels, remap
            src_label = split_lbls / (img_path.stem + ".txt")
            new_lines = []
            if src_label.exists():
                for line in src_label.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    src_id = int(parts[0])
                    new_id = class_map.get(src_id)
                    if new_id is None:
                        continue
                    new_lines.append(f"{new_id} " + " ".join(parts[1:]))

            # Run model, take landmark detections as pseudo labels
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            result = det.detect(img, confidence=0.3)
            n_lm = 0
            for landmark_class in (
                "board-conner", "palace-bottom", "palace-center",
                "palace-conner", "board-border",
            ):
                cls_id = V6_CLASSES.index(landmark_class)
                for lm in result.get_landmarks(landmark_class):
                    if lm.confidence < args.landmark_conf:
                        continue
                    x1, y1, x2, y2 = lm.bbox
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    n_lm += 1
            n_landmarks_added += n_lm

            # Write out (prefix with split to avoid name collisions)
            stem = f"{split}_{img_path.stem}"
            shutil.copy(img_path, out_imgs / f"{stem}{img_path.suffix}")
            (out_lbls / f"{stem}.txt").write_text("\n".join(new_lines))

    print(f"\n  Processed {n_total} images, added {n_landmarks_added} landmark pseudo-labels")

    # Cleanup
    shutil.rmtree(src_dir)

    print(f"\n[4/4] Done. Output: data/{args.name}/train/")
    print(f"\nNext: split + train:")
    print(f"  python scripts/split_items.py --dir data/{args.name}")
    print(f"  python scripts/train_items.py --data data/{args.name}/data.yaml --name {args.name}")


if __name__ == "__main__":
    main()
