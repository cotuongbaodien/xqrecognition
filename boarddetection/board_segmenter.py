"""Board segmentation: locate the Xiangqi board outline via a YOLO-seg model.

Returns the 4 grid corners of the board. More robust than landmark-point
detection for tilted/perspective boards because the segmentation model sees
the whole board region, not sparse intersection points.
"""

from typing import Optional, Tuple, List

import cv2
import numpy as np


class BoardSegmenter:
    """Segments the board and extracts its 4 grid corners."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def get_board_quad(
        self, image: np.ndarray, confidence: float = 0.25
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        """Return (tl, tr, bl, br) image-space corners of the board, or None.

        Picks the largest detected mask, reduces its polygon to 4 corners
        (approxPolyDP, falling back to minAreaRect), and orders them by
        (x±y) extremes.
        """
        if self.model is None:
            return None
        result = self.model(image, conf=confidence, verbose=False)[0]
        if result.masks is None or len(result.masks) == 0:
            return None

        polys = result.masks.xy
        if not polys:
            return None
        # Largest mask = the board (ignore spurious small masks)
        biggest = max(polys, key=lambda p: cv2.contourArea(p.astype(np.float32)))
        biggest = biggest.astype(np.float32)

        # Reduce to 4 corners
        quad = self._reduce_to_quad(biggest)
        if quad is None:
            return None

        pts = [(float(p[0]), float(p[1])) for p in quad]
        tl = min(pts, key=lambda p: p[0] + p[1])
        br = max(pts, key=lambda p: p[0] + p[1])
        tr = max(pts, key=lambda p: p[0] - p[1])
        bl = min(pts, key=lambda p: p[0] - p[1])
        if len({tl, tr, bl, br}) != 4:
            return None
        return tl, tr, bl, br

    @staticmethod
    def _reduce_to_quad(contour: np.ndarray) -> Optional[np.ndarray]:
        """Reduce a polygon contour to exactly 4 vertices."""
        peri = cv2.arcLength(contour, True)
        for eps_frac in (0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10):
            approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
            if len(approx) == 4:
                return approx.reshape(-1, 2)
        # Fallback: rotated bounding rectangle (always 4 points)
        rect = cv2.minAreaRect(contour)
        return cv2.boxPoints(rect)
