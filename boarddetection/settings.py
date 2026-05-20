"""
Configuration settings for boarddetection package.
"""

from pathlib import Path

# Package paths (boarddetection/ folder)
PACKAGE_ROOT = Path(__file__).parent
MODELS_DIR = PACKAGE_ROOT / "models"

# Grid dimensions for Xiangqi board
GRID_COLS = 9   # 0-8 (files a-i)
GRID_ROWS = 10  # 0-9 (ranks)
TOTAL_INTERSECTIONS = GRID_COLS * GRID_ROWS  # 90

# Board segmentation class
BOARD_SEG_CLASSES = {
    0: "inters"  # Intersection points
}

# Piece detection classes - mapping class ID to (name, FEN symbol)
# =============================================================
# Item detection classes (NEW STANDARD - itemdetection.yolov8)
# 18 classes: 14 pieces + 4 landmarks
# Used by models/items.pt
# =============================================================
ITEM_CLASSES = {
    # IDs match Roboflow v6+ alphabetical ordering — board-border at 7 shifts
    # other landmarks +1 from old (v3-v5) numbering.
    0:  ("black-advisor",  "a"),
    1:  ("black-cannon",   "c"),
    2:  ("black-chariot",  "r"),
    3:  ("black-elephant", "b"),
    4:  ("black-general",  "k"),
    5:  ("black-horse",    "n"),
    6:  ("black-soldier",  "p"),
    7:  ("board-border",   None),   # NEW v6+: 26 perimeter grid points
    8:  ("board-conner",   None),
    9:  ("palace-bottom",  None),
    10: ("palace-center",  None),
    11: ("palace-conner",  None),
    12: ("red-advisor",    "A"),
    13: ("red-cannon",     "C"),
    14: ("red-chariot",    "R"),
    15: ("red-elephant",   "B"),
    16: ("red-general",    "K"),
    17: ("red-horse",      "N"),
    18: ("red-soldier",    "P"),
}

# Piece / landmark ID sets for ItemDetector
PIECE_CLASS_IDS = {cid for cid, (_, fen) in ITEM_CLASSES.items() if fen is not None}
LANDMARK_CLASS_IDS = {cid for cid, (_, fen) in ITEM_CLASSES.items() if fen is None}
ITEM_CLASS_NAMES = [ITEM_CLASSES[i][0] for i in range(len(ITEM_CLASSES))]

# Landmark class name → ID
LANDMARK_NAMES = {
    "board-border":  7,
    "board-conner":  8,
    "palace-bottom": 9,
    "palace-center": 10,
    "palace-conner": 11,
}

# =============================================================
# Legacy piece classes (LEGACY - for models/pieces_det.pt)
# 14 classes only, original Roboflow training order
# =============================================================
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

# Class ID to FEN symbol mapping (legacy pieces_det.pt)
CLASS_TO_FEN = {cid: fen for cid, (_, fen) in PIECE_CLASSES.items()}

# FEN symbol to class ID mapping (legacy)
FEN_TO_CLASS = {fen: cid for cid, (_, fen) in PIECE_CLASSES.items()}

# Class names list (legacy)
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

# Display names for visualization (Vietnamese) - supports BOTH naming conventions
PIECE_DISPLAY_NAMES = {
    # New (kebab-case)
    "black-advisor":  "Sĩ đen",
    "red-advisor":    "Sĩ đỏ",
    "black-cannon":   "Pháo đen",
    "red-cannon":     "Pháo đỏ",
    "black-elephant": "Tượng đen",
    "red-elephant":   "Tượng đỏ",
    "black-general":  "Tướng đen",
    "red-general":    "Tướng đỏ",
    "black-horse":    "Mã đen",
    "red-horse":      "Mã đỏ",
    "black-soldier":  "Tốt đen",
    "red-soldier":    "Tốt đỏ",
    "black-chariot":  "Xe đen",
    "red-chariot":    "Xe đỏ",
    # Legacy (snake_case)
    "Advisor_black":  "Sĩ đen",
    "Advisor_red":    "Sĩ đỏ",
    "Cannon_black":   "Pháo đen",
    "Cannon_red":     "Pháo đỏ",
    "Elephant_black": "Tượng đen",
    "Elephant_red":   "Tượng đỏ",
    "General_black":  "Tướng đen",
    "General_red":    "Tướng đỏ",
    "Knight_black":   "Mã đen",
    "Knight_red":     "Mã đỏ",
    "Pawn_black":     "Tốt đen",
    "Pawn_red":       "Tốt đỏ",
    "Rook_black":     "Xe đen",
    "Rook_red":       "Xe đỏ",
}

# Models
ITEMS_MODEL = MODELS_DIR / "items.pt"  # Unified model: pieces + landmarks
