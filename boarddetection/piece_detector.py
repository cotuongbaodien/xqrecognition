"""
Chess piece detection module for Xiangqi Recognition System.
Detects and classifies the 14 types of Xiangqi pieces.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .settings import (
    PIECE_CLASSES,
    CLASS_TO_FEN,
    PIECE_CONFIDENCE_THRESHOLD,
    PIECE_DISPLAY_NAMES,
)


@dataclass
class DetectedPiece:
    """Represents a detected chess piece."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]
    fen_symbol: str

    @property
    def display_name(self) -> str:
        """Get the Vietnamese display name for the piece."""
        return PIECE_DISPLAY_NAMES.get(self.class_name, self.class_name)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "display_name": self.display_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": self.center,
            "fen_symbol": self.fen_symbol,
        }


class PieceDetector:
    """
    Detects and classifies Xiangqi chess pieces.
    Uses YOLOv8 detection model trained on 14 piece classes.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the PieceDetector.

        Args:
            model_path: Path to the trained YOLOv8 detection model.
        """
        self.model = None
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the YOLOv8 detection model."""
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect_pieces(
        self,
        image: np.ndarray,
        confidence: float = PIECE_CONFIDENCE_THRESHOLD
    ) -> List[DetectedPiece]:
        """
        Detect chess pieces in the image.

        Args:
            image: Input image as numpy array (BGR).
            confidence: Minimum confidence threshold.

        Returns:
            List of detected pieces.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference
        results = self.model(image, conf=confidence, verbose=False)

        detected_pieces = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Get bounding box coordinates
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = bbox

                # Get class ID and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())

                # Get class name and FEN symbol
                if class_id in PIECE_CLASSES:
                    class_name, fen_symbol = PIECE_CLASSES[class_id]
                else:
                    # Unknown class
                    class_name = f"Unknown_{class_id}"
                    fen_symbol = "?"

                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                piece = DetectedPiece(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    center=(float(center_x), float(center_y)),
                    fen_symbol=fen_symbol,
                )
                detected_pieces.append(piece)

        return detected_pieces

    def get_piece_center(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """
        Calculate the center point of a bounding box.

        Args:
            bbox: Bounding box (x1, y1, x2, y2).

        Returns:
            Center point (x, y).
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def filter_by_confidence(
        self,
        pieces: List[DetectedPiece],
        min_confidence: float
    ) -> List[DetectedPiece]:
        """Filter pieces by minimum confidence threshold."""
        return [p for p in pieces if p.confidence >= min_confidence]

    def filter_by_class(
        self,
        pieces: List[DetectedPiece],
        class_ids: List[int]
    ) -> List[DetectedPiece]:
        """Filter pieces by class IDs."""
        return [p for p in pieces if p.class_id in class_ids]

    def get_red_pieces(self, pieces: List[DetectedPiece]) -> List[DetectedPiece]:
        """Get only red pieces (uppercase FEN symbols)."""
        return [p for p in pieces if p.fen_symbol.isupper()]

    def get_black_pieces(self, pieces: List[DetectedPiece]) -> List[DetectedPiece]:
        """Get only black pieces (lowercase FEN symbols)."""
        return [p for p in pieces if p.fen_symbol.islower()]

    def visualize_detections(
        self,
        image: np.ndarray,
        pieces: List[DetectedPiece],
        show_confidence: bool = True,
        show_label: bool = True
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on the image.

        Args:
            image: Input image as numpy array (BGR).
            pieces: List of detected pieces.
            show_confidence: Whether to show confidence scores.
            show_label: Whether to show class labels.

        Returns:
            Image with detections drawn.
        """
        result = image.copy()

        # Colors for red and black pieces
        red_color = (0, 0, 255)  # BGR
        black_color = (0, 0, 0)  # BGR
        text_bg = (255, 255, 255)

        for piece in pieces:
            x1, y1, x2, y2 = map(int, piece.bbox)

            # Determine color based on piece color
            color = red_color if piece.fen_symbol.isupper() else black_color

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cx, cy = map(int, piece.center)
            cv2.circle(result, (cx, cy), 4, color, -1)

            if show_label:
                # Prepare label text
                label_parts = [piece.fen_symbol]
                if show_confidence:
                    label_parts.append(f"{piece.confidence:.2f}")
                label = " ".join(label_parts)

                # Get text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw text background
                cv2.rectangle(
                    result,
                    (x1, y1 - text_height - 5),
                    (x1 + text_width + 5, y1),
                    text_bg,
                    -1
                )

                # Draw text
                cv2.putText(
                    result,
                    label,
                    (x1 + 2, y1 - 5),
                    font,
                    font_scale,
                    color,
                    thickness
                )

        return result

    def non_max_suppression(
        self,
        pieces: List[DetectedPiece],
        iou_threshold: float = 0.5
    ) -> List[DetectedPiece]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            pieces: List of detected pieces.
            iou_threshold: IoU threshold for suppression.

        Returns:
            Filtered list of pieces.
        """
        if not pieces:
            return []

        # Sort by confidence (highest first)
        pieces = sorted(pieces, key=lambda p: p.confidence, reverse=True)

        keep = []
        while pieces:
            # Keep the piece with highest confidence
            best = pieces.pop(0)
            keep.append(best)

            # Filter out pieces with high IoU overlap
            pieces = [
                p for p in pieces
                if self._compute_iou(best.bbox, p.bbox) < iou_threshold
            ]

        return keep

    def _compute_iou(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]
    ) -> float:
        """Compute Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Compute intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 < xi1 or yi2 < yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
