"""
Board detection module for Xiangqi Recognition System.
Detects board bounding box and constructs the 9x10 grid.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    GRID_COLS,
    GRID_ROWS,
    TOTAL_INTERSECTIONS,
    BOARD_CONFIDENCE_THRESHOLD,
)


@dataclass
class Point:
    """Represents a 2D point."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: "Point") -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Grid:
    """Represents the 9x10 Xiangqi board grid."""
    points: np.ndarray  # Shape: (10, 9, 2) - [row][col][x,y]
    cell_width: float
    cell_height: float

    def get_point(self, row: int, col: int) -> Optional[Point]:
        """Get the intersection point at given row and column."""
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            x, y = self.points[row, col]
            return Point(x, y)
        return None

    def get_nearest_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Find the nearest grid cell for a given point.

        Returns:
            Tuple of (row, col) for the nearest intersection.
        """
        min_dist = float('inf')
        nearest_row, nearest_col = 0, 0

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                px, py = self.points[row, col]
                dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_row, nearest_col = row, col

        return nearest_row, nearest_col


class BoardDetector:
    """
    Detects the Xiangqi board using YOLOv8-seg model.
    Used as a fallback when BoardBoxDetector fails.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the YOLOv8 segmentation model."""
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect_intersections(
        self,
        image: np.ndarray,
        confidence: float = BOARD_CONFIDENCE_THRESHOLD
    ) -> List[Point]:
        """
        Detect intersection points in the image.

        Args:
            image: Input image as numpy array (BGR).
            confidence: Minimum confidence threshold.

        Returns:
            List of detected intersection points.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model(image, conf=confidence, verbose=False)
        points = []

        for result in results:
            if result.masks is None:
                continue

            for mask in result.masks.xy:
                if len(mask) > 0:
                    centroid_x = np.mean(mask[:, 0])
                    centroid_y = np.mean(mask[:, 1])
                    points.append(Point(centroid_x, centroid_y))

        return points

    def build_grid(self, intersections: List[Point]) -> Optional[Grid]:
        """
        Build the 9x10 grid from detected intersection points.

        Args:
            intersections: List of detected intersection points.

        Returns:
            Grid object if successful, None otherwise.
        """
        if len(intersections) < TOTAL_INTERSECTIONS:
            print(f"Warning: Only {len(intersections)} intersections detected, expected {TOTAL_INTERSECTIONS}")

        points_array = np.array([[p.x, p.y] for p in intersections])
        sorted_by_y = points_array[points_array[:, 1].argsort()]

        grid_points = np.zeros((GRID_ROWS, GRID_COLS, 2))
        points_per_row = len(sorted_by_y) // GRID_ROWS
        row_points = []

        for i in range(GRID_ROWS):
            start_idx = i * points_per_row
            end_idx = start_idx + points_per_row if i < GRID_ROWS - 1 else len(sorted_by_y)

            row = sorted_by_y[start_idx:end_idx]
            row = row[row[:, 0].argsort()]

            if len(row) >= GRID_COLS:
                row = row[:GRID_COLS]
            else:
                row = self._interpolate_row(row, GRID_COLS)

            row_points.append(row)

        for row_idx, row in enumerate(row_points):
            for col_idx in range(min(len(row), GRID_COLS)):
                grid_points[row_idx, col_idx] = row[col_idx]

        cell_widths = []
        cell_heights = []

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS - 1):
                width = grid_points[row, col + 1, 0] - grid_points[row, col, 0]
                if width > 0:
                    cell_widths.append(width)

        for row in range(GRID_ROWS - 1):
            for col in range(GRID_COLS):
                height = grid_points[row + 1, col, 1] - grid_points[row, col, 1]
                if height > 0:
                    cell_heights.append(height)

        cell_width = np.mean(cell_widths) if cell_widths else 0
        cell_height = np.mean(cell_heights) if cell_heights else 0

        return Grid(points=grid_points, cell_width=cell_width, cell_height=cell_height)

    def _interpolate_row(self, row: np.ndarray, target_cols: int) -> np.ndarray:
        """Interpolate a row to have exactly target_cols points."""
        if len(row) == 0:
            return np.zeros((target_cols, 2))

        if len(row) == 1:
            return np.tile(row, (target_cols, 1))

        x_vals = np.linspace(row[0, 0], row[-1, 0], target_cols)
        y_vals = np.interp(x_vals, row[:, 0], row[:, 1])

        return np.column_stack([x_vals, y_vals])

    def build_grid_from_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        margin: float = 0.02
    ) -> Grid:
        """
        Build grid from board bounding box.

        Args:
            bbox: Bounding box (x1, y1, x2, y2) of the board.
            margin: Optional margin to shrink the bbox (as fraction of size).

        Returns:
            Grid object with calculated intersection points.
        """
        x1, y1, x2, y2 = bbox

        if margin > 0:
            w, h = x2 - x1, y2 - y1
            x1 += w * margin
            y1 += h * margin
            x2 -= w * margin
            y2 -= h * margin

        board_width = x2 - x1
        board_height = y2 - y1

        cell_width = board_width / (GRID_COLS - 1)
        cell_height = board_height / (GRID_ROWS - 1)

        grid_points = np.zeros((GRID_ROWS, GRID_COLS, 2))

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                grid_points[row, col, 0] = x1 + col * cell_width
                grid_points[row, col, 1] = y1 + row * cell_height

        return Grid(points=grid_points, cell_width=cell_width, cell_height=cell_height)

    def visualize_grid(
        self,
        image: np.ndarray,
        grid: Grid,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1
    ) -> np.ndarray:
        """Draw the grid on the image."""
        result = image.copy()

        # Draw horizontal lines
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS - 1):
                pt1 = tuple(map(int, grid.points[row, col]))
                pt2 = tuple(map(int, grid.points[row, col + 1]))
                cv2.line(result, pt1, pt2, color, thickness)

        # Draw vertical lines
        for col in range(GRID_COLS):
            for row in range(GRID_ROWS - 1):
                pt1 = tuple(map(int, grid.points[row, col]))
                pt2 = tuple(map(int, grid.points[row + 1, col]))
                cv2.line(result, pt1, pt2, color, thickness)

        # Draw intersection points
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                pt = tuple(map(int, grid.points[row, col]))
                cv2.circle(result, pt, 3, (0, 0, 255), -1)

        return result


class BoardBoxDetector:
    """
    Detects the Xiangqi board bounding box using YOLOv8 detection model.
    Primary method for board detection.
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the YOLOv8 detection model."""
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect_board(
        self,
        image: np.ndarray,
        confidence: float = BOARD_CONFIDENCE_THRESHOLD
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Detect the board bounding box in the image.

        Args:
            image: Input image as numpy array (BGR).
            confidence: Minimum confidence threshold.

        Returns:
            Bounding box (x1, y1, x2, y2) or None if not detected.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model(image, conf=confidence, verbose=False)

        best_box = None
        best_conf = 0.0

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    best_box = (float(x1), float(y1), float(x2), float(y2))

        return best_box

    def build_grid_from_detection(
        self,
        image: np.ndarray,
        confidence: float = BOARD_CONFIDENCE_THRESHOLD,
        margin: float = 0.0
    ) -> Optional[Grid]:
        """
        Detect board and build grid in one step.

        Args:
            image: Input image as numpy array (BGR).
            confidence: Minimum confidence threshold.
            margin: Optional margin to shrink the bbox.

        Returns:
            Grid object or None if board not detected.
        """
        bbox = self.detect_board(image, confidence)

        if bbox is None:
            return None

        board_detector = BoardDetector()
        return board_detector.build_grid_from_bbox(bbox, margin)
