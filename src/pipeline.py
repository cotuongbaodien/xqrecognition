"""
Main detection pipeline for Xiangqi Recognition System.
Combines board detection, piece detection, and FEN generation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BOARD_SEG_MODEL,
    PIECES_DET_MODEL,
    BOARD_CONFIDENCE_THRESHOLD,
    PIECE_CONFIDENCE_THRESHOLD,
)
from .board_detector import BoardDetector, Grid, Point
from .piece_detector import PieceDetector, DetectedPiece
from .fen_generator import FENGenerator, BoardState


@dataclass
class RecognitionResult:
    """Result of the recognition pipeline."""
    fen: str
    board_state: BoardState
    pieces: List[DetectedPiece]
    grid: Optional[Grid]
    image_shape: Tuple[int, int, int]
    confidence: float
    visualization: Optional[np.ndarray] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "fen": self.fen,
            "pieces": [p.to_dict() for p in self.pieces],
            "piece_count": len(self.pieces),
            "board_state": {
                "pieces": self.board_state.pieces,
                "counts": self.board_state.count_pieces(),
            },
            "image_shape": self.image_shape,
            "confidence": self.confidence,
            "errors": self.errors,
        }


class XiangqiRecognizer:
    """
    Main recognition pipeline for Xiangqi board images.

    Combines:
    - Board detection and grid construction
    - Chess piece detection
    - FEN notation generation
    """

    def __init__(
        self,
        board_model_path: str = None,
        pieces_model_path: str = None,
        use_board_detection: bool = True
    ):
        """
        Initialize the Xiangqi recognizer.

        Args:
            board_model_path: Path to the board segmentation model.
            pieces_model_path: Path to the pieces detection model.
            use_board_detection: Whether to use board detection for grid construction.
                If False, uses interpolation-based grid estimation.
        """
        self.use_board_detection = use_board_detection

        # Initialize detectors
        self.board_detector = BoardDetector()
        self.piece_detector = PieceDetector()
        self.fen_generator = FENGenerator()

        # Load models if paths provided
        board_path = board_model_path or str(BOARD_SEG_MODEL)
        pieces_path = pieces_model_path or str(PIECES_DET_MODEL)

        if Path(board_path).exists():
            self.board_detector.load_model(board_path)
        else:
            print(f"Warning: Board model not found at {board_path}")
            self.use_board_detection = False

        if Path(pieces_path).exists():
            self.piece_detector.load_model(pieces_path)
        else:
            raise FileNotFoundError(f"Pieces model not found at {pieces_path}")

    def recognize(
        self,
        image_path: str,
        board_confidence: float = BOARD_CONFIDENCE_THRESHOLD,
        piece_confidence: float = PIECE_CONFIDENCE_THRESHOLD,
        visualize: bool = False
    ) -> RecognitionResult:
        """
        Recognize the Xiangqi board from an image file.

        Args:
            image_path: Path to the input image.
            board_confidence: Confidence threshold for board detection.
            piece_confidence: Confidence threshold for piece detection.
            visualize: Whether to generate visualization image.

        Returns:
            RecognitionResult object.
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return self.recognize_image(
            image,
            board_confidence=board_confidence,
            piece_confidence=piece_confidence,
            visualize=visualize
        )

    def recognize_image(
        self,
        image: np.ndarray,
        board_confidence: float = BOARD_CONFIDENCE_THRESHOLD,
        piece_confidence: float = PIECE_CONFIDENCE_THRESHOLD,
        visualize: bool = False
    ) -> RecognitionResult:
        """
        Recognize the Xiangqi board from an image array.

        Args:
            image: Input image as numpy array (BGR).
            board_confidence: Confidence threshold for board detection.
            piece_confidence: Confidence threshold for piece detection.
            visualize: Whether to generate visualization image.

        Returns:
            RecognitionResult object.
        """
        errors = []
        grid = None
        h, w = image.shape[:2]

        # Step 1: Detect pieces
        pieces = self.piece_detector.detect_pieces(image, confidence=piece_confidence)

        # Apply NMS to remove duplicate detections
        pieces = self.piece_detector.non_max_suppression(pieces, iou_threshold=0.5)

        # Step 2: Build grid
        if self.use_board_detection and self.board_detector.model is not None:
            # Detect intersections
            intersections = self.board_detector.detect_intersections(
                image, confidence=board_confidence
            )

            if len(intersections) >= 20:  # Minimum points for reasonable grid
                grid = self.board_detector.build_grid(intersections)
            else:
                errors.append(f"Only {len(intersections)} intersections detected")

        # Step 3: Map pieces to grid
        if grid is not None:
            board_state = self.fen_generator.map_pieces_to_grid(pieces, grid)
        else:
            # Fallback: use interpolation-based mapping
            board_state = self.fen_generator.map_pieces_to_grid_by_interpolation(
                pieces, w, h
            )
            errors.append("Using interpolation-based grid estimation")

        # Step 4: Normalize board orientation (ensure black at top, red at bottom)
        board_state = self.fen_generator.normalize_board_orientation(board_state)

        # Step 5: Generate FEN
        fen = self.fen_generator.generate_fen(board_state)

        # Calculate overall confidence
        if pieces:
            avg_confidence = sum(p.confidence for p in pieces) / len(pieces)
        else:
            avg_confidence = 0.0

        # Step 6: Generate visualization if requested
        visualization = None
        if visualize:
            visualization = self._create_visualization(image, pieces, grid, board_state)

        return RecognitionResult(
            fen=fen,
            board_state=board_state,
            pieces=pieces,
            grid=grid,
            image_shape=image.shape,
            confidence=avg_confidence,
            visualization=visualization,
            errors=errors,
        )

    def _create_visualization(
        self,
        image: np.ndarray,
        pieces: List[DetectedPiece],
        grid: Optional[Grid],
        board_state: BoardState
    ) -> np.ndarray:
        """Create a visualization of the detection results."""
        vis = image.copy()

        # Draw grid if available
        if grid is not None:
            vis = self.board_detector.visualize_grid(vis, grid)

        # Draw piece detections
        vis = self.piece_detector.visualize_detections(vis, pieces)

        # Add FEN text at the bottom
        fen = self.fen_generator.generate_fen(board_state)
        h, w = vis.shape[:2]

        # Create a bar for FEN display
        bar_height = 30
        vis = cv2.copyMakeBorder(vis, 0, bar_height, 0, 0, cv2.BORDER_CONSTANT, value=(50, 50, 50))

        # Add FEN text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, f"FEN: {fen}", (10, h + 22), font, 0.5, (255, 255, 255), 1)

        return vis

    def recognize_batch(
        self,
        image_paths: List[str],
        **kwargs
    ) -> List[RecognitionResult]:
        """
        Recognize multiple images.

        Args:
            image_paths: List of image paths.
            **kwargs: Additional arguments passed to recognize().

        Returns:
            List of RecognitionResult objects.
        """
        results = []
        for path in image_paths:
            try:
                result = self.recognize(path, **kwargs)
                results.append(result)
            except Exception as e:
                # Create error result
                result = RecognitionResult(
                    fen="",
                    board_state=BoardState(
                        board=[[None] * 9 for _ in range(10)],
                        pieces=[]
                    ),
                    pieces=[],
                    grid=None,
                    image_shape=(0, 0, 0),
                    confidence=0.0,
                    errors=[str(e)],
                )
                results.append(result)

        return results


def create_recognizer(
    board_model: str = None,
    pieces_model: str = None,
    use_board_detection: bool = True
) -> XiangqiRecognizer:
    """
    Factory function to create a XiangqiRecognizer.

    Args:
        board_model: Path to board segmentation model.
        pieces_model: Path to pieces detection model.
        use_board_detection: Whether to use board detection.

    Returns:
        Configured XiangqiRecognizer instance.
    """
    return XiangqiRecognizer(
        board_model_path=board_model,
        pieces_model_path=pieces_model,
        use_board_detection=use_board_detection
    )
