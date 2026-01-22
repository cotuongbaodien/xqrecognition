"""
Validate prepared data for auto-learning pipeline.
Usage: python scripts/auto_learn/validate.py
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "prepare"
IMAGES_DIR = DATA_DIR / "images"
LABELS_FILE = DATA_DIR / "labels.csv"


def validate_fen(fen: str) -> tuple[bool, str]:
    """Validate FEN string format."""
    rows = fen.split("/")

    if len(rows) != 10:
        return False, f"FEN phải có 10 hàng, hiện có {len(rows)}"

    valid_pieces = set("rnbakcp RNBAKCP")
    generals = {"k": 0, "K": 0}

    for i, row in enumerate(rows):
        col_count = 0
        for char in row:
            if char.isdigit():
                col_count += int(char)
            elif char in valid_pieces:
                col_count += 1
                if char in generals:
                    generals[char] += 1
            else:
                return False, f"Ký tự không hợp lệ '{char}' ở hàng {i}"

        if col_count != 9:
            return False, f"Hàng {i} có {col_count} cột, cần 9"

    if generals["k"] != 1:
        return False, f"Cần đúng 1 tướng đen (k), hiện có {generals['k']}"
    if generals["K"] != 1:
        return False, f"Cần đúng 1 tướng đỏ (K), hiện có {generals['K']}"

    return True, "OK"


def validate_data():
    """Validate all data in prepare folder."""
    print("=" * 60)
    print("VALIDATE DATA - Auto Learn Pipeline")
    print("=" * 60)

    # Check folders exist
    if not DATA_DIR.exists():
        print(f"\n[ERROR] Folder không tồn tại: {DATA_DIR}")
        return False

    if not IMAGES_DIR.exists():
        print(f"\n[ERROR] Folder images không tồn tại: {IMAGES_DIR}")
        return False

    if not LABELS_FILE.exists():
        print(f"\n[ERROR] File labels.csv không tồn tại: {LABELS_FILE}")
        return False

    # Count images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images_in_folder = {
        f.name for f in IMAGES_DIR.iterdir()
        if f.suffix.lower() in image_extensions
    }

    print(f"\n[INFO] Tìm thấy {len(images_in_folder)} ảnh trong folder images/")

    # Read labels.csv
    errors = []
    warnings = []
    valid_count = 0
    images_in_csv = set()

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):  # Start from 2 (after header)
            image_name = row.get("image", "").strip()
            fen = row.get("fen", "").strip()

            if not image_name:
                errors.append(f"Dòng {row_num}: Thiếu tên ảnh")
                continue

            if not fen:
                errors.append(f"Dòng {row_num}: Thiếu FEN cho {image_name}")
                continue

            images_in_csv.add(image_name)

            # Check image exists
            if image_name not in images_in_folder:
                errors.append(f"Dòng {row_num}: Ảnh '{image_name}' không tồn tại trong folder images/")
                continue

            # Validate FEN
            is_valid, msg = validate_fen(fen)
            if not is_valid:
                errors.append(f"Dòng {row_num} ({image_name}): FEN lỗi - {msg}")
                continue

            valid_count += 1

    # Check for images without labels
    images_without_labels = images_in_folder - images_in_csv
    if images_without_labels:
        for img in sorted(images_without_labels):
            if not img.startswith("example_"):  # Ignore example files
                warnings.append(f"Ảnh '{img}' chưa có FEN trong labels.csv")

    # Print results
    print(f"\n[RESULTS]")
    print(f"  - Ảnh trong folder: {len(images_in_folder)}")
    print(f"  - Entries trong CSV: {len(images_in_csv)}")
    print(f"  - Hợp lệ: {valid_count}")
    print(f"  - Lỗi: {len(errors)}")
    print(f"  - Cảnh báo: {len(warnings)}")

    if errors:
        print(f"\n[ERRORS] ({len(errors)})")
        for err in errors[:20]:  # Show first 20
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... và {len(errors) - 20} lỗi khác")

    if warnings:
        print(f"\n[WARNINGS] ({len(warnings)})")
        for warn in warnings[:10]:  # Show first 10
            print(f"  - {warn}")
        if len(warnings) > 10:
            print(f"  ... và {len(warnings) - 10} cảnh báo khác")

    # Final status
    print("\n" + "=" * 60)
    if errors:
        print("[FAILED] Có lỗi cần sửa trước khi train")
        return False
    elif valid_count == 0:
        print("[FAILED] Chưa có data hợp lệ")
        return False
    else:
        print(f"[PASSED] Sẵn sàng train với {valid_count} samples")
        return True


if __name__ == "__main__":
    success = validate_data()
    sys.exit(0 if success else 1)
