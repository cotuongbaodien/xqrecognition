"""
Xiangqi Recognition System - Source Package.

This package contains the core modules for:
- Board detection and grid construction
- Chess piece detection
- FEN notation generation
- Main recognition pipeline
"""

from .board_detector import BoardDetector
from .piece_detector import PieceDetector
from .fen_generator import FENGenerator
from .pipeline import XiangqiRecognizer

__all__ = [
    "BoardDetector",
    "PieceDetector",
    "FENGenerator",
    "XiangqiRecognizer",
]
