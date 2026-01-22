"""
Board detection module for Xiangqi Recognition System.
Detects intersection points on the board and constructs the 9x10 grid.
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
    Detects the Xiangqi board and constructs the grid.
    Uses YOLOv8-seg model to detect intersection points.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the BoardDetector.

        Args:
            model_path: Path to the trained YOLOv8-seg model.
        """
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

        # Run inference
        results = self.model(image, conf=confidence, verbose=False)

        points = []

        for result in results:
            if result.masks is None:
                continue

            # Extract mask polygons and compute centroids
            for mask in result.masks.xy:
                if len(mask) > 0:
                    # Calculate centroid of the polygon
                    centroid_x = np.mean(mask[:, 0])
                    centroid_y = np.mean(mask[:, 1])
                    points.append(Point(centroid_x, centroid_y))

        return points

    def build_grid(self, intersections: List[Point]) -> Optional[Grid]:
        """
        Build the 9x10 grid from detected intersection points.

        Algorithm:
        1. Sort points by y-coordinate to group into rows
        2. Within each row, sort by x-coordinate
        3. Verify we have exactly 90 points (9x10)

        Args:
            intersections: List of detected intersection points.

        Returns:
            Grid object if successful, None otherwise.
        """
        if len(intersections) < TOTAL_INTERSECTIONS:
            print(f"Warning: Only {len(intersections)} intersections detected, expected {TOTAL_INTERSECTIONS}")

        # Convert to numpy array for easier processing
        points_array = np.array([[p.x, p.y] for p in intersections])

        # Sort by y-coordinate first (top to bottom)
        sorted_by_y = points_array[points_array[:, 1].argsort()]

        # Group into rows using clustering
        grid_points = np.zeros((GRID_ROWS, GRID_COLS, 2))

        # Use k-means-like clustering to group points into rows
        points_per_row = len(sorted_by_y) // GRID_ROWS
        row_points = []

        for i in range(GRID_ROWS):
            start_idx = i * points_per_row
            end_idx = start_idx + points_per_row if i < GRID_ROWS - 1 else len(sorted_by_y)

            # Get points for this row
            row = sorted_by_y[start_idx:end_idx]

            # Sort by x-coordinate (left to right)
            row = row[row[:, 0].argsort()]

            # Take exactly GRID_COLS points
            if len(row) >= GRID_COLS:
                row = row[:GRID_COLS]
            else:
                # Pad with interpolated points if needed
                row = self._interpolate_row(row, GRID_COLS)

            row_points.append(row)

        # Build grid array
        for row_idx, row in enumerate(row_points):
            for col_idx in range(min(len(row), GRID_COLS)):
                grid_points[row_idx, col_idx] = row[col_idx]

        # Calculate average cell dimensions
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

        # Simple linear interpolation
        x_vals = np.linspace(row[0, 0], row[-1, 0], target_cols)
        y_vals = np.interp(x_vals, row[:, 0], row[:, 1])

        return np.column_stack([x_vals, y_vals])

    def build_grid_from_corners(
        self,
        corners: List[Point],
        image_shape: Tuple[int, int]
    ) -> Grid:
        """
        Build grid from four corner points using perspective transformation.
        This is a fallback method when intersection detection doesn't work well.

        Args:
            corners: Four corner points [top-left, top-right, bottom-right, bottom-left]
            image_shape: (height, width) of the image

        Returns:
            Grid object
        """
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        corners_array = np.array([[p.x, p.y] for p in corners], dtype=np.float32)

        # Compute grid points by interpolation
        grid_points = np.zeros((GRID_ROWS, GRID_COLS, 2))

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                # Bilinear interpolation
                u = col / (GRID_COLS - 1)
                v = row / (GRID_ROWS - 1)

                # Interpolate along top and bottom edges
                top = corners_array[0] * (1 - u) + corners_array[1] * u
                bottom = corners_array[3] * (1 - u) + corners_array[2] * u

                # Interpolate between top and bottom
                point = top * (1 - v) + bottom * v
                grid_points[row, col] = point

        # Calculate cell dimensions
        cell_width = np.mean([
            grid_points[0, 1, 0] - grid_points[0, 0, 0],
            grid_points[-1, 1, 0] - grid_points[-1, 0, 0]
        ])
        cell_height = np.mean([
            grid_points[1, 0, 1] - grid_points[0, 0, 1],
            grid_points[1, -1, 1] - grid_points[0, -1, 1]
        ])

        return Grid(points=grid_points, cell_width=cell_width, cell_height=cell_height)

    def detect_board_corners(self, image: np.ndarray) -> Optional[List[Point]]:
        """
        Detect the four corners of the board using traditional CV methods.
        This is a fallback method.

        Args:
            image: Input image as numpy array (BGR).

        Returns:
            List of four corner points, or None if detection fails.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest quadrilateral contour
        max_area = 0
        best_approx = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                continue

            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if it's a quadrilateral
            if len(approx) == 4 and area > max_area:
                max_area = area
                best_approx = approx

        if best_approx is None:
            return None

        # Convert to Point objects
        corners = [Point(float(p[0][0]), float(p[0][1])) for p in best_approx]

        # Sort corners: top-left, top-right, bottom-right, bottom-left
        corners = self._sort_corners(corners)

        return corners

    def _sort_corners(self, corners: List[Point]) -> List[Point]:
        """Sort corners in order: top-left, top-right, bottom-right, bottom-left."""
        # Calculate centroid
        cx = sum(p.x for p in corners) / 4
        cy = sum(p.y for p in corners) / 4

        # Classify corners based on position relative to centroid
        top_left = None
        top_right = None
        bottom_left = None
        bottom_right = None

        for p in corners:
            if p.x < cx and p.y < cy:
                top_left = p
            elif p.x >= cx and p.y < cy:
                top_right = p
            elif p.x < cx and p.y >= cy:
                bottom_left = p
            else:
                bottom_right = p

        # Handle edge cases where corners might be on the same side
        if None in [top_left, top_right, bottom_left, bottom_right]:
            # Fallback: sort by sum and difference of coordinates
            sorted_corners = sorted(corners, key=lambda p: p.x + p.y)
            top_left = sorted_corners[0]
            bottom_right = sorted_corners[3]

            sorted_corners = sorted(corners, key=lambda p: p.x - p.y)
            top_right = sorted_corners[3]
            bottom_left = sorted_corners[0]

        return [top_left, top_right, bottom_right, bottom_left]

    def visualize_grid(
        self,
        image: np.ndarray,
        grid: Grid,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1
    ) -> np.ndarray:
        """
        Draw the grid on the image.

        Args:
            image: Input image as numpy array (BGR).
            grid: Grid object to visualize.
            color: Line color in BGR format.
            thickness: Line thickness.

        Returns:
            Image with grid drawn.
        """
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
