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
    BOARD_DET_MODEL,
    PIECES_DET_MODEL,
    MODELS_DIR,
    BOARD_CONFIDENCE_THRESHOLD,
    PIECE_CONFIDENCE_THRESHOLD,
)
from .board_detector import BoardDetector, BoardBoxDetector, LandmarkDetector, Grid
from .piece_detector import PieceDetector, DetectedPiece
from .item_detector import ItemDetector
from .fen_generator import FENGenerator, BoardState
from .rules_validator import RulesValidator


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
        board_det_model_path: str = None,
        pieces_model_path: str = None,
        use_board_detection: bool = True
    ):
        """
        Initialize the Xiangqi recognizer.

        Args:
            board_model_path: Path to the board segmentation model (intersections).
            board_det_model_path: Path to the board detection model (bounding box).
            pieces_model_path: Path to the pieces detection model.
            use_board_detection: Whether to use board detection for grid construction.
        """
        self.use_board_detection = use_board_detection

        # Initialize detectors
        self.board_detector = BoardDetector()
        self.board_box_detector = BoardBoxDetector()
        self.landmark_detector = LandmarkDetector()
        self.item_detector = ItemDetector()
        self.piece_detector = PieceDetector()
        self.fen_generator = FENGenerator()
        self.rules_validator = RulesValidator()

        # Load models
        board_seg_path = board_model_path or str(BOARD_SEG_MODEL)
        board_det_path = board_det_model_path or str(BOARD_DET_MODEL)
        pieces_path = pieces_model_path or str(PIECES_DET_MODEL)
        landmarks_path = str(MODELS_DIR / "landmarks.pt")
        items_path = str(MODELS_DIR / "items.pt")

        # Load item model (primary for landmarks)
        if Path(items_path).exists():
            self.item_detector.load_model(items_path)
            print(f"Loaded item model from {items_path}")

        # Load landmark model (legacy fallback)
        if Path(landmarks_path).exists():
            self.landmark_detector.load_model(landmarks_path)

        # Load board segmentation model (fallback)
        if Path(board_seg_path).exists():
            self.board_detector.load_model(board_seg_path)

        # Load board detection model (fallback)
        if Path(board_det_path).exists():
            self.board_box_detector.load_model(board_det_path)

        # Load pieces model (required)
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
        bbox = None
        h, w = image.shape[:2]

        # Step 1: Detect pieces (using legacy pieces_det.pt - more piece data)
        pieces = self.piece_detector.detect_pieces(image, confidence=piece_confidence)
        pieces = self.piece_detector.non_max_suppression(pieces, iou_threshold=0.35)

        # Cap at max 32 pieces (maximum in Xiangqi)
        if len(pieces) > 32:
            pieces = sorted(pieces, key=lambda p: p.confidence, reverse=True)[:32]

        # Step 2: Detect items (landmarks) for orientation/mirror later
        item_result = None
        if self.item_detector.model is not None:
            item_result = self.item_detector.detect(image, confidence=0.3)

        # Step 2a: Build grid from board box detection (most reliable for piece mapping)
        if self.use_board_detection and self.board_box_detector.model is not None:
            bbox = self.board_box_detector.detect_board(image, confidence=board_confidence)
            if bbox is not None:
                grid = self.board_detector.build_grid_from_bbox(bbox)
            else:
                errors.append("Board bounding box not detected")

        # Step 2b: Fallback to item landmarks if board_det fails
        if grid is None and item_result is not None:
            grid = ItemDetector.build_grid_from_landmarks(
                item_result, image_shape=(h, w)
            )
            if grid is None:
                errors.append("Item detector: failed to build grid from landmarks")

        # Step 2c: Fallback to YOLO intersection detection
        if grid is None and self.board_detector.model is not None:
            intersections = self.board_detector.detect_intersections(
                image, confidence=board_confidence
            )
            if len(intersections) >= 50:
                grid = self.board_detector.build_grid(intersections)
            else:
                errors.append(f"Only {len(intersections)} intersections detected")

        # Step 3: Map pieces to grid
        if grid is not None:
            board_state = self.fen_generator.map_pieces_to_grid(pieces, grid)
        else:
            board_state = self.fen_generator.map_pieces_to_grid_by_interpolation(
                pieces, w, h
            )
            errors.append("Using interpolation-based grid estimation")

        # Step 4: Normalize VERTICAL orientation only (red at bottom, black at top)
        # We do NOT apply horizontal mirror - the consuming app handles mirror.
        # Mirror auto-detection is unreliable and can produce wrong results.
        orientation = self.fen_generator.detect_board_orientation(board_state)
        if orientation == 'flipped':
            board_state = self.fen_generator.flip_board(board_state)

        # Step 5: Validate and correct using game rules
        piece_confidences = {}
        for piece in pieces:
            cx, cy = piece.center
            if grid is not None:
                row, col = grid.get_nearest_cell(cx, cy)
            else:
                # Approximate row/col for interpolation case
                centers_x = [p.center[0] for p in pieces]
                centers_y = [p.center[1] for p in pieces]
                min_x, max_x = min(centers_x), max(centers_x)
                min_y, max_y = min(centers_y), max(centers_y)
                cell_w = (max_x - min_x) / 8 if max_x > min_x else 1
                cell_h = (max_y - min_y) / 9 if max_y > min_y else 1
                col = round((cx - min_x) / cell_w)
                row = round((cy - min_y) / cell_h)
                col = max(0, min(8, col))
                row = max(0, min(9, row))
            piece_confidences[(row, col)] = piece.confidence
        board_state = self.rules_validator.validate_and_correct(
            board_state, piece_confidences
        )

        # Step 6: Generate FEN
        fen = self.fen_generator.generate_fen(board_state)

        # Calculate confidence
        avg_confidence = sum(p.confidence for p in pieces) / len(pieces) if pieces else 0.0

        # Step 6: Generate visualization
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

        if grid is not None:
            vis = self.board_detector.visualize_grid(vis, grid)

        vis = self.piece_detector.visualize_detections(vis, pieces)

        # Add FEN text
        fen = self.fen_generator.generate_fen(board_state)
        h, w = vis.shape[:2]

        bar_height = 30
        vis = cv2.copyMakeBorder(vis, 0, bar_height, 0, 0, cv2.BORDER_CONSTANT, value=(50, 50, 50))

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
    board_det_model: str = None,
    pieces_model: str = None,
    use_board_detection: bool = True
) -> XiangqiRecognizer:
    """
    Factory function to create a XiangqiRecognizer.

    Args:
        board_model: Path to board segmentation model.
        board_det_model: Path to board detection model (bounding box).
        pieces_model: Path to pieces detection model.
        use_board_detection: Whether to use board detection.

    Returns:
        Configured XiangqiRecognizer instance.
    """
    return XiangqiRecognizer(
        board_model_path=board_model,
        board_det_model_path=board_det_model,
        pieces_model_path=pieces_model,
        use_board_detection=use_board_detection
    )
