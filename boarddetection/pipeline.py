"""
Main detection pipeline for Xiangqi Recognition System.
Combines board detection, piece detection, and FEN generation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

from .settings import (
    ITEMS_MODEL,
    MODELS_DIR,
    PIECE_CONFIDENCE_THRESHOLD,
)
from .board_detector import BoardDetector, Grid
from .board_segmenter import BoardSegmenter
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

    Uses a single unified model (items.pt) that detects both pieces (14 classes)
    and board landmarks (4 classes) in one forward pass.
    """

    def __init__(self, items_model_path: str = None, **_legacy_kwargs):
        """
        Initialize the Xiangqi recognizer.

        Args:
            items_model_path: Path to the unified items detection model.
                              Defaults to models/items.pt.
        """
        self.item_detector = ItemDetector()
        self.piece_detector = PieceDetector()  # utility (NMS + visualization only)
        self.board_detector = BoardDetector()  # utility (grid drawing only)
        self.fen_generator = FENGenerator()
        self.rules_validator = RulesValidator()

        items_path = items_model_path or str(ITEMS_MODEL)
        if not Path(items_path).exists():
            raise FileNotFoundError(f"Items model not found at {items_path}")
        self.item_detector.load_model(items_path)
        print(f"Loaded items model from {items_path}")

        # Optional board-segmentation model for robust board localization
        # (preferred over landmark-point fitting when available).
        self.board_segmenter = None
        seg_path = MODELS_DIR / "board_seg.pt"
        if seg_path.exists():
            self.board_segmenter = BoardSegmenter(str(seg_path))
            print(f"Loaded board-seg model from {seg_path}")

    def recognize(
        self,
        image_path: str,
        piece_confidence: float = PIECE_CONFIDENCE_THRESHOLD,
        visualize: bool = False,
        **_legacy
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
            piece_confidence=piece_confidence,
            visualize=visualize
        )

    def recognize_image(
        self,
        image: np.ndarray,
        piece_confidence: float = PIECE_CONFIDENCE_THRESHOLD,
        visualize: bool = False,
        **_legacy
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
        h, w = image.shape[:2]

        # Single forward pass: pieces + landmarks from items.pt
        item_result = self.item_detector.detect(image, confidence=piece_confidence)

        pieces = self.piece_detector.non_max_suppression(
            item_result.pieces, iou_threshold=0.35
        )
        # Cap at max 32 pieces (maximum in Xiangqi)
        if len(pieces) > 32:
            pieces = sorted(pieces, key=lambda p: p.confidence, reverse=True)[:32]

        # Build grid. Prefer board-segmentation (robust to tilt/perspective),
        # refining its 4 corners with detected board-conner landmarks (which
        # sit AT grid corners by definition, so they give pixel-precise
        # positions on clean boards). Fall back to landmark-point fitting if
        # seg unavailable or fails.
        grid = None
        if self.board_segmenter is not None:
            quad = self.board_segmenter.get_board_quad(image)
            if quad is not None:
                quad = self._snap_quad_to_corners(
                    quad, item_result.board_corners
                )
                grid = ItemDetector.build_grid_from_quad(
                    quad, pieces,
                    palace_centers=item_result.palace_centers,
                    palace_corners=item_result.palace_corners,
                    palace_bottoms=item_result.palace_bottoms,
                    image_shape=(h, w),
                )
                if grid is None:
                    errors.append("Board-seg quad found but grid build failed")
        if grid is None:
            grid = ItemDetector.build_grid_from_landmarks(
                item_result, image_shape=(h, w)
            )
        if grid is None:
            errors.append("Failed to build grid")

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
            visualization = self._create_visualization(
                image, pieces, grid, board_state, item_result
            )

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

    @staticmethod
    def _snap_quad_to_corners(quad, board_corners, threshold=50.0):
        """Refine segmentation quad corners using detected board-conner
        landmarks. Each board-conner sits AT a grid corner by definition, so
        when one lies near a seg corner, it gives a pixel-precise position.
        Greedy 1-to-1 match prevents two detections collapsing to one corner.
        Seg corners with no nearby detection keep their value (robust on
        tilted boards where corners aren't detected)."""
        if not board_corners:
            return quad
        slots = list(quad)  # [tl, tr, bl, br]
        bc_pts = [l.center for l in board_corners]
        taken = set()
        for bc in bc_pts:
            order = sorted(
                range(4),
                key=lambda i: (bc[0] - slots[i][0]) ** 2 + (bc[1] - slots[i][1]) ** 2,
            )
            for i in order:
                if i in taken:
                    continue
                d = ((bc[0] - slots[i][0]) ** 2 + (bc[1] - slots[i][1]) ** 2) ** 0.5
                if d < threshold:
                    slots[i] = (float(bc[0]), float(bc[1]))
                    taken.add(i)
                break
        return tuple(slots)

    def _create_visualization(
        self,
        image: np.ndarray,
        pieces: List[DetectedPiece],
        grid: Optional[Grid],
        board_state: BoardState,
        item_result=None,
    ) -> np.ndarray:
        """Create a visualization of the detection results."""
        vis = image.copy()

        if grid is not None:
            vis = self.board_detector.visualize_grid(vis, grid)

        vis = self.piece_detector.visualize_detections(vis, pieces)

        # Draw landmark detections with distinct colors per class
        if item_result is not None:
            landmark_colors = {
                "board-conner":  (255, 255, 0),    # cyan
                "palace-center": (255, 0, 255),    # magenta
                "palace-conner": (0, 255, 255),    # yellow
                "palace-bottom": (255, 128, 0),    # orange
                "board-border":  (128, 255, 128),  # light green (v6+)
            }
            font = cv2.FONT_HERSHEY_SIMPLEX
            for name, color in landmark_colors.items():
                for lm in item_result.get_landmarks(name):
                    x1, y1, x2, y2 = map(int, lm.bbox)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{name[:4]}.{name.split('-')[1][:3]} {lm.confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
                    cv2.rectangle(vis, (x1, y2), (x1 + tw + 4, y2 + th + 6),
                                  color, -1)
                    cv2.putText(vis, label, (x1 + 2, y2 + th + 2),
                                font, 0.4, (0, 0, 0), 1)

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


def create_recognizer(items_model: str = None) -> XiangqiRecognizer:
    """Factory function to create a XiangqiRecognizer."""
    return XiangqiRecognizer(items_model_path=items_model)
