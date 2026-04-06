"""
Sinh YOLO labels từ ảnh có FEN ground truth.
Dùng board_det.pt để detect bàn cờ, sau đó dùng FEN để tạo bounding box cho từng quân.

Workflow:
1. Detect board bounding box (board_det.pt)
2. Parse FEN → biết quân nào ở ô nào
3. Tính tọa độ từng ô trên grid 9x10
4. Tạo bounding box YOLO cho mỗi quân
5. Lưu label file + copy ảnh vào dataset

Usage:
    python scripts/generate_yolo_labels_from_fen.py
    python scripts/generate_yolo_labels_from_fen.py --input data/fendata --output data/pieces_merged --split train
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import GRID_COLS, GRID_ROWS, FEN_TO_CLASS
from src.board_detector import BoardBoxDetector


def parse_fen_to_board(fen: str):
    """Parse FEN string thành board 10x9."""
    # Lấy phần board (trước space)
    board_str = fen.strip().split(" ")[0]
    rows = board_str.split("/")

    board = []
    for row_str in rows:
        row = []
        for char in row_str:
            if char.isdigit():
                row.extend([None] * int(char))
            elif char.isalpha():
                row.append(char)
        # Pad hoặc trim to 9 columns
        row = row[:GRID_COLS]
        while len(row) < GRID_COLS:
            row.append(None)
        board.append(row)

    # Đảm bảo 10 rows
    while len(board) < GRID_ROWS:
        board.append([None] * GRID_COLS)

    return board[:GRID_ROWS]


def generate_yolo_labels(
    board,
    bbox,
    img_width: int,
    img_height: int,
    piece_size_ratio: float = 0.06
):
    """
    Sinh YOLO labels từ board state và board bounding box.

    Args:
        board: Board 10x9 từ parse_fen_to_board
        bbox: (x1, y1, x2, y2) bounding box của bàn cờ
        img_width: Chiều rộng ảnh
        img_height: Chiều cao ảnh
        piece_size_ratio: Kích thước quân cờ (tỉ lệ so với cell size)

    Returns:
        List of (class_id, cx_norm, cy_norm, w_norm, h_norm) YOLO format
    """
    x1, y1, x2, y2 = bbox
    board_w = x2 - x1
    board_h = y2 - y1

    cell_w = board_w / (GRID_COLS - 1)
    cell_h = board_h / (GRID_ROWS - 1)

    # Kích thước bounding box cho quân cờ
    piece_w = cell_w * 0.85
    piece_h = cell_h * 0.85

    labels = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            piece = board[row][col]
            if piece is None:
                continue

            class_id = FEN_TO_CLASS.get(piece)
            if class_id is None:
                continue

            # Tọa độ tâm quân cờ trên ảnh
            cx = x1 + col * cell_w
            cy = y1 + row * cell_h

            # Normalize theo kích thước ảnh
            cx_norm = cx / img_width
            cy_norm = cy / img_height
            w_norm = piece_w / img_width
            h_norm = piece_h / img_height

            # Clamp values
            cx_norm = max(0.0, min(1.0, cx_norm))
            cy_norm = max(0.0, min(1.0, cy_norm))
            w_norm = min(w_norm, min(cx_norm, 1.0 - cx_norm) * 2)
            h_norm = min(h_norm, min(cy_norm, 1.0 - cy_norm) * 2)

            labels.append((class_id, cx_norm, cy_norm, w_norm, h_norm))

    return labels


def process_dataset(
    input_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    split: str = "train",
    prefix: str = "fen_",
    board_confidence: float = 0.3
):
    """
    Xử lý toàn bộ dataset: detect board + sinh YOLO labels.

    Args:
        input_dir: Thư mục chứa ảnh
        labels_csv: File CSV với columns (image, fen)
        output_dir: Thư mục output (pieces_merged format)
        split: train/valid/test
        prefix: Prefix cho tên file (tránh trùng)
        board_confidence: Confidence threshold cho board detection
    """
    # Load board detector
    board_det = BoardBoxDetector()
    board_det_path = PROJECT_ROOT / "models" / "board_det.pt"
    if not board_det_path.exists():
        print(f"Lỗi: Không tìm thấy model {board_det_path}")
        return

    board_det.load_model(str(board_det_path))
    print(f"Đã load board detection model: {board_det_path}")

    # Load FEN labels
    fen_map = {}
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen_map[row["image"]] = row["fen"]

    print(f"Đã load {len(fen_map)} FEN labels từ {labels_csv}")

    # Output directories
    out_images = output_dir / split / "images"
    out_labels = output_dir / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    skipped = 0

    for img_name, fen in fen_map.items():
        img_path = input_dir / img_name
        if not img_path.exists():
            skipped += 1
            continue

        # Đọc ảnh
        image = cv2.imread(str(img_path))
        if image is None:
            failed += 1
            continue

        h, w = image.shape[:2]

        # Detect board bounding box
        bbox = board_det.detect_board(image, confidence=board_confidence)
        if bbox is None:
            print(f"  Không detect được board: {img_name}")
            failed += 1
            continue

        # Parse FEN
        try:
            board = parse_fen_to_board(fen)
        except Exception as e:
            print(f"  FEN lỗi ({img_name}): {e}")
            failed += 1
            continue

        # Sinh YOLO labels
        labels = generate_yolo_labels(board, bbox, w, h)
        if not labels:
            failed += 1
            continue

        # Lưu
        out_name = f"{prefix}{Path(img_name).stem}"
        out_img_path = out_images / f"{out_name}{img_path.suffix}"
        out_lbl_path = out_labels / f"{out_name}.txt"

        shutil.copy(img_path, out_img_path)

        with open(out_lbl_path, "w") as f:
            for class_id, cx, cy, bw, bh in labels:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        success += 1

    print(f"\nKết quả:")
    print(f"  Thành công: {success}")
    print(f"  Thất bại: {failed}")
    print(f"  Bỏ qua (không tìm thấy ảnh): {skipped}")
    print(f"  Output: {output_dir / split}")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Sinh YOLO labels từ ảnh có FEN ground truth"
    )
    parser.add_argument(
        "--input", type=str, default="data/fendata",
        help="Thư mục chứa images/ và labels.csv"
    )
    parser.add_argument(
        "--output", type=str, default="data/pieces_merged",
        help="Thư mục output dataset"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Split: train/valid/test"
    )
    parser.add_argument(
        "--prefix", type=str, default="fen_",
        help="Prefix cho tên file"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Board detection confidence threshold"
    )

    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output
    labels_csv = input_dir / "labels.csv"
    images_dir = input_dir / "images"

    if not images_dir.exists():
        print(f"Lỗi: Không tìm thấy {images_dir}")
        return

    if not labels_csv.exists():
        print(f"Lỗi: Không tìm thấy {labels_csv}")
        return

    print("=" * 60)
    print("Sinh YOLO Labels Từ FEN Ground Truth")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Split: {args.split}")

    success = process_dataset(
        images_dir, labels_csv, output_dir,
        split=args.split,
        prefix=args.prefix,
        board_confidence=args.confidence
    )

    if success:
        # Cập nhật data.yaml nếu cần
        print(f"\nĐã thêm {success} ảnh vào {output_dir / args.split}")
        print("Chạy lại training để sử dụng data mới.")


if __name__ == "__main__":
    main()
