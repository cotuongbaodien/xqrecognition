"""boarddetection — Xiangqi (Chinese chess) board recognition.

Public API:

    from boarddetection import XiangqiRecognizer

    recognizer = XiangqiRecognizer()  # uses bundled models/items.pt
    result = recognizer.recognize("board.jpg")
    print(result.fen)
"""

from .pipeline import XiangqiRecognizer, RecognitionResult, create_recognizer
from .piece_detector import DetectedPiece
from .item_detector import Landmark, ItemDetectionResult
from .board_detector import Grid
from .fen_generator import BoardState, render_fen_ascii
from .settings import ITEMS_MODEL, PIECE_DISPLAY_NAMES, ITEM_CLASSES

__all__ = [
    "XiangqiRecognizer",
    "RecognitionResult",
    "create_recognizer",
    "DetectedPiece",
    "Landmark",
    "ItemDetectionResult",
    "Grid",
    "BoardState",
    "render_fen_ascii",
    "ITEMS_MODEL",
    "PIECE_DISPLAY_NAMES",
    "ITEM_CLASSES",
]
