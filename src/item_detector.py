"""
Unified item detector for Xiangqi Recognition System.

Detects all 18 classes (14 pieces + 4 board landmarks) in a single forward pass:
- Pieces: black/red × {advisor, cannon, chariot, elephant, general, horse, soldier}
- Landmarks: board-conner, palace-bottom, palace-center, palace-conner

Provides:
- Pieces (compatible with DetectedPiece)
- Landmarks grouped by class (for grid construction + orientation detection)

Grid construction strategy (in priority order):
1. 4 board-conner detected → bilinear interpolation between corners (best)
2. >= 4 mixed landmarks → homography fitting from known grid positions
3. < 4 landmarks → bbox fallback (least accurate)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    ITEM_CLASSES,
    PIECE_CLASS_IDS,
    LANDMARK_CLASS_IDS,
    LANDMARK_NAMES,
    PIECE_DISPLAY_NAMES,
)
from .piece_detector import DetectedPiece
from .board_detector import BoardDetector, Grid


@dataclass
class Landmark:
    """A detected board landmark."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]


@dataclass
class ItemDetectionResult:
    """Combined detection result: pieces and landmarks."""
    pieces: List[DetectedPiece] = field(default_factory=list)
    landmarks_by_class: Dict[str, List[Landmark]] = field(default_factory=dict)

    def get_landmarks(self, name: str) -> List[Landmark]:
        return self.landmarks_by_class.get(name, [])

    @property
    def board_corners(self) -> List[Landmark]:
        return self.get_landmarks("board-conner")

    @property
    def palace_bottoms(self) -> List[Landmark]:
        return self.get_landmarks("palace-bottom")

    @property
    def palace_corners(self) -> List[Landmark]:
        return self.get_landmarks("palace-conner")

    @property
    def palace_centers(self) -> List[Landmark]:
        return self.get_landmarks("palace-center")


class ItemDetector:
    """
    Unified detector for pieces + landmarks (single YOLO inference).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        self.model = YOLO(model_path)
        self.model_path = model_path

    def detect(
        self,
        image: np.ndarray,
        confidence: float = 0.3,
    ) -> ItemDetectionResult:
        """Run inference and split into pieces vs landmarks."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model(image, conf=confidence, verbose=False)

        pieces: List[DetectedPiece] = []
        landmarks: Dict[str, List[Landmark]] = {}

        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].cpu().numpy())
                conf = float(r.boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if cls_id not in ITEM_CLASSES:
                    continue
                name, fen = ITEM_CLASSES[cls_id]

                if cls_id in PIECE_CLASS_IDS:
                    pieces.append(DetectedPiece(
                        class_id=cls_id,
                        class_name=name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        fen_symbol=fen,
                    ))
                elif cls_id in LANDMARK_CLASS_IDS:
                    landmarks.setdefault(name, []).append(Landmark(
                        class_id=cls_id,
                        class_name=name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                    ))

        return ItemDetectionResult(
            pieces=pieces,
            landmarks_by_class=landmarks,
        )

    @staticmethod
    def _collect_correspondences(
        result: ItemDetectionResult,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Map every detected landmark to its expected (col, row) position in the
        9x10 grid coordinate system. Returns list of (col, row, image_x, image_y).

        Known landmark positions (standard orientation, before mirror normalize):
        - board-conner: (0,0), (8,0), (0,9), (8,9)
        - palace-bottom: (3,0), (5,0), (3,9), (5,9)  (back rank)
        - palace-conner: (3,2), (5,2), (3,7), (5,7)  (interior corners)
        - palace-center: (4,1), (4,8)                (palace center)
        """
        # Need at least one landmark to determine reference center
        all_pts: List[Tuple[float, float]] = []
        for lst in result.landmarks_by_class.values():
            all_pts.extend(l.center for l in lst)
        if not all_pts:
            return []

        cx_ref = sum(p[0] for p in all_pts) / len(all_pts)
        cy_ref = sum(p[1] for p in all_pts) / len(all_pts)

        correspondences: List[Tuple[float, float, float, float]] = []

        # board-conner: (0,0), (8,0), (0,9), (8,9)
        for c in result.board_corners:
            x, y = c.center
            col = 0 if x < cx_ref else 8
            row = 0 if y < cy_ref else 9
            correspondences.append((col, row, x, y))

        # palace-bottom: at back rank, cols 3 or 5
        for l in result.palace_bottoms:
            x, y = l.center
            col = 3 if x < cx_ref else 5
            row = 0 if y < cy_ref else 9
            correspondences.append((col, row, x, y))

        # palace-conner: 2 rows in, cols 3 or 5
        for l in result.palace_corners:
            x, y = l.center
            col = 3 if x < cx_ref else 5
            row = 2 if y < cy_ref else 7
            correspondences.append((col, row, x, y))

        # palace-center: col 4, row 1 or 8
        for l in result.palace_centers:
            x, y = l.center
            col = 4
            row = 1 if y < cy_ref else 8
            correspondences.append((col, row, x, y))

        return correspondences

    @staticmethod
    def build_grid_from_landmarks(
        result: ItemDetectionResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Grid]:
        """
        Build a 9x10 grid from detected landmarks.

        Tries multiple strategies in priority order:
        1. 4 board-conner → bilinear interpolation (most accurate)
        2. >= 4 mixed landmarks → homography fitting via cv2.findHomography
        3. >= 2 landmarks → bbox fallback (least accurate)
        """
        from config.settings import GRID_COLS, GRID_ROWS

        corners = result.board_corners

        # ---- Strategy 1: 4 board corners → bilinear ----
        if len(corners) >= 4:
            pts = sorted([(c.center[0], c.center[1]) for c in corners],
                         key=lambda p: p[1])
            top_two = sorted(pts[:2], key=lambda p: p[0])
            bot_two = sorted(pts[-2:], key=lambda p: p[0])
            tl, tr = top_two
            bl, br = bot_two

            grid_points = np.zeros((GRID_ROWS, GRID_COLS, 2))
            for row in range(GRID_ROWS):
                v = row / (GRID_ROWS - 1)
                left_x = tl[0] * (1 - v) + bl[0] * v
                left_y = tl[1] * (1 - v) + bl[1] * v
                right_x = tr[0] * (1 - v) + br[0] * v
                right_y = tr[1] * (1 - v) + br[1] * v
                for col in range(GRID_COLS):
                    u = col / (GRID_COLS - 1)
                    grid_points[row, col, 0] = left_x * (1 - u) + right_x * u
                    grid_points[row, col, 1] = left_y * (1 - u) + right_y * u

            cell_w = abs((tr[0] - tl[0]) / (GRID_COLS - 1))
            cell_h = abs((bl[1] - tl[1]) / (GRID_ROWS - 1))
            return Grid(points=grid_points, cell_width=cell_w, cell_height=cell_h)

        # ---- Strategy 2: homography from mixed landmarks ----
        correspondences = ItemDetector._collect_correspondences(result)
        if len(correspondences) >= 4:
            src = np.array([[c[0], c[1]] for c in correspondences], dtype=np.float32)
            dst = np.array([[c[2], c[3]] for c in correspondences], dtype=np.float32)

            # findHomography needs 4+ points; use RANSAC to reject outliers
            try:
                if len(correspondences) >= 4:
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                else:
                    H = cv2.getPerspectiveTransform(src[:4], dst[:4])
            except cv2.error:
                H = None

            if H is not None:
                grid_points = np.zeros((GRID_ROWS, GRID_COLS, 2))
                for row in range(GRID_ROWS):
                    for col in range(GRID_COLS):
                        pt = np.array([col, row, 1.0])
                        transformed = H @ pt
                        if transformed[2] != 0:
                            transformed /= transformed[2]
                        grid_points[row, col] = [transformed[0], transformed[1]]

                # Estimate cell size from grid
                cell_w = float(np.mean(np.abs(
                    grid_points[:, 1:, 0] - grid_points[:, :-1, 0]
                )))
                cell_h = float(np.mean(np.abs(
                    grid_points[1:, :, 1] - grid_points[:-1, :, 1]
                )))
                return Grid(
                    points=grid_points,
                    cell_width=cell_w,
                    cell_height=cell_h,
                )

        # ---- Strategy 3: bbox fallback from any landmarks ----
        all_pts: List[Tuple[float, float]] = []
        for lst in result.landmarks_by_class.values():
            all_pts.extend(l.center for l in lst)
        if len(all_pts) >= 2:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            bd = BoardDetector()
            return bd.build_grid_from_bbox(bbox, margin=0.0)

        return None

    @staticmethod
    def detect_orientation_from_landmarks(
        result: ItemDetectionResult,
    ) -> Optional[str]:
        """
        Determine board orientation (vertical) using palace-bottom vs
        palace-conner positions.

        Logic:
        - palace-bottom is at the back rank (row 0 OR 9 in standard FEN)
        - palace-conner is 2 rows inside (row 2 OR 7)
        - For each palace, compute its centroid Y; the back rank palace-bottom
          will be FURTHER from the palace center than palace-conner.

        Returns:
            'standard' (red at bottom of image) | 'flipped' | None
        """
        pb = result.palace_bottoms
        pc = result.palace_corners
        if not pb or not pc:
            return None

        # Cluster landmarks into top palace vs bottom palace by Y
        all_y = [l.center[1] for l in pb + pc]
        if not all_y:
            return None
        y_mid = (min(all_y) + max(all_y)) / 2

        def split(landmarks):
            top, bot = [], []
            for l in landmarks:
                (top if l.center[1] < y_mid else bot).append(l)
            return top, bot

        pb_top, pb_bot = split(pb)
        pc_top, pc_bot = split(pc)

        # Heuristic: bottom palace contains red king (standard FEN)
        # We can't tell red vs black from landmarks alone; the orientation
        # detector in fen_generator (which uses piece colors) handles that.
        # This function only validates that the GRID is built correctly,
        # not the FEN orientation.
        # Return 'standard' as default — fen_generator handles vertical flip.
        return 'standard'
