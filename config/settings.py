"""
Configuration settings for Xiangqi Recognition System.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DOWNLOAD_DIR = PROJECT_ROOT / "download"

# Dataset paths
BOARD_SEG_DATA = DATA_DIR / "board_seg"
PIECES_DATA = DATA_DIR / "pieces"

# Model paths
BOARD_SEG_MODEL = MODELS_DIR / "board_seg.pt"    # Board segmentation (fallback)
BOARD_DET_MODEL = MODELS_DIR / "board_det.pt"    # Board bounding box (primary)
PIECES_DET_MODEL = MODELS_DIR / "pieces_det.pt"  # Pieces detection

# Dataset zip files
BOARD_SEG_ZIP = DOWNLOAD_DIR / "seg_chinese_chess.v3i.yolov8.zip"
PIECES_DET_ZIP = DOWNLOAD_DIR / "Chinese-chess.v9i.yolov8.zip"

# Grid dimensions for Xiangqi board
GRID_COLS = 9   # 0-8 (files a-i)
GRID_ROWS = 10  # 0-9 (ranks)
TOTAL_INTERSECTIONS = GRID_COLS * GRID_ROWS  # 90

# Board segmentation class
BOARD_SEG_CLASSES = {
    0: "inters"  # Intersection points
}

# Piece detection classes - mapping class ID to (name, FEN symbol)
PIECE_CLASSES = {
    0: ("Advisor_black", "a"),
    1: ("Advisor_red", "A"),
    2: ("Cannon_black", "c"),
    3: ("Cannon_red", "C"),
    4: ("Elephant_black", "b"),
    5: ("Elephant_red", "B"),
    6: ("General_black", "k"),
    7: ("General_red", "K"),
    8: ("Knight_black", "n"),
    9: ("Knight_red", "N"),
    10: ("Pawn_black", "p"),
    11: ("Pawn_red", "P"),
    12: ("Rook_black", "r"),
    13: ("Rook_red", "R"),
}

# Class ID to FEN symbol mapping
CLASS_TO_FEN = {class_id: fen for class_id, (_, fen) in PIECE_CLASSES.items()}

# FEN symbol to class ID mapping
FEN_TO_CLASS = {fen: class_id for class_id, (_, fen) in PIECE_CLASSES.items()}

# Class names list (for YOLO training)
PIECE_CLASS_NAMES = [PIECE_CLASSES[i][0] for i in range(len(PIECE_CLASSES))]

# Training settings
TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "patience": 20,
    "device": "auto",  # auto-detect: cuda, mps, or cpu
}

# Board segmentation training settings
BOARD_SEG_TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 8,
    "img_size": 640,
    "patience": 20,
    "device": "auto",
}

# Detection confidence thresholds
BOARD_CONFIDENCE_THRESHOLD = 0.5
PIECE_CONFIDENCE_THRESHOLD = 0.5

# Standard Xiangqi starting position FEN
STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"

# Display names for visualization (Vietnamese)
PIECE_DISPLAY_NAMES = {
    "Advisor_black": "Sĩ đen",
    "Advisor_red": "Sĩ đỏ",
    "Cannon_black": "Pháo đen",
    "Cannon_red": "Pháo đỏ",
    "Elephant_black": "Tượng đen",
    "Elephant_red": "Tượng đỏ",
    "General_black": "Tướng đen",
    "General_red": "Tướng đỏ",
    "Knight_black": "Mã đen",
    "Knight_red": "Mã đỏ",
    "Pawn_black": "Tốt đen",
    "Pawn_red": "Tốt đỏ",
    "Rook_black": "Xe đen",
    "Rook_red": "Xe đỏ",
}
