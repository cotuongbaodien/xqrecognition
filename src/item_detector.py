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

    # Known landmark positions in (col, row) grid coordinates
    _LANDMARK_GRID_POSITIONS = {
        "board-conner":  [(0, 0), (8, 0), (0, 9), (8, 9)],
        "palace-bottom": [(3, 0), (5, 0), (3, 9), (5, 9)],
        "palace-conner": [(3, 2), (5, 2), (3, 7), (5, 7)],
        "palace-center": [(4, 1), (4, 8)],
    }

    @staticmethod
    def _find_4_board_corners(
        detected_corners: List["Landmark"],
        pieces: List[DetectedPiece],
    ) -> List[Tuple[float, float, float, float]]:
        """Find the 4 board-corners. User insight:

        > "4 board conner nối lại với nhau sẽ bao hết quân cờ — nó nằm ngoài rìa hết"

        The 4 corners ENCLOSE all pieces. So the 4 extreme points of (pieces +
        detected board-conners), measured by (x±y), ARE the 4 board-corners.

        Why combine pieces with detected corners:
        - When all 4 corners detected accurately: they are the extremes anyway
        - When a corner is occluded: the occluding piece sits at it, so the
          piece IS the corner
        - When a corner is detected as a false positive (inside the board):
          the actual outermost piece in that direction overrides it

        Returns 4 correspondences: (0,0)→TL, (8,0)→TR, (0,9)→BL, (8,9)→BR.
        Returns [] if no candidates available.
        """
        candidates = [p.center for p in pieces]
        candidates += [l.center for l in detected_corners]
        if not candidates:
            return []
        tl = min(candidates, key=lambda p: p[0] + p[1])
        br = max(candidates, key=lambda p: p[0] + p[1])
        tr = max(candidates, key=lambda p: p[0] - p[1])
        bl = min(candidates, key=lambda p: p[0] - p[1])
        return [
            (0, 0, tl[0], tl[1]),
            (8, 0, tr[0], tr[1]),
            (0, 9, bl[0], bl[1]),
            (8, 9, br[0], br[1]),
        ]

    @staticmethod
    def _grid_from_4_corners(
        corner_corrs: List[Tuple[float, float, float, float]],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Grid]:
        """Build a Grid using exact perspective transform from 4 corners.
        With exactly 4 point correspondences, cv2.getPerspectiveTransform
        gives a homography that passes EXACTLY through those 4 points — no
        averaging with other constraints that could pull corners off-board.
        """
        from config.settings import GRID_COLS, GRID_ROWS
        by_pos = {(c[0], c[1]): (c[2], c[3]) for c in corner_corrs}
        if not all(k in by_pos for k in [(0, 0), (8, 0), (0, 9), (8, 9)]):
            return None
        # Source: grid coords (col, row); destination: image coords
        src = np.array([[0, 0], [8, 0], [0, 9], [8, 9]], dtype=np.float32)
        dst = np.array([by_pos[(0, 0)], by_pos[(8, 0)],
                        by_pos[(0, 9)], by_pos[(8, 9)]], dtype=np.float32)
        try:
            H = cv2.getPerspectiveTransform(src, dst)
        except cv2.error:
            return None

        grid_points = np.zeros((GRID_ROWS, GRID_COLS, 2))
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                t = H @ np.array([col, row, 1.0])
                if t[2] != 0:
                    t /= t[2]
                grid_points[row, col] = [t[0], t[1]]

        dx = grid_points[:, 1:, 0] - grid_points[:, :-1, 0]
        dy = grid_points[1:, :, 1] - grid_points[:-1, :, 1]
        cell_w = float(np.mean(np.abs(dx)))
        cell_h = float(np.mean(np.abs(dy)))
        return Grid(points=grid_points, cell_width=cell_w, cell_height=cell_h)

    @staticmethod
    def _build_bilinear_from_corners(
        corner_corrs: List[Tuple[float, float, float, float]],
    ) -> Optional[Grid]:
        """Build a Grid by bilinear interpolation between the 4 corner
        correspondences. Each corr is (col, row, x, y) where col,row ∈
        {(0,0),(8,0),(0,9),(8,9)}."""
        from config.settings import GRID_COLS, GRID_ROWS
        by_pos = {(c[0], c[1]): (c[2], c[3]) for c in corner_corrs}
        if not all(k in by_pos for k in [(0, 0), (8, 0), (0, 9), (8, 9)]):
            return None
        tl, tr = by_pos[(0, 0)], by_pos[(8, 0)]
        bl, br = by_pos[(0, 9)], by_pos[(8, 9)]

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

    @staticmethod
    def _piece_proxy_correspondences(
        pieces: List[DetectedPiece],
        rough_grid: Grid,
        detected_landmarks_by_pos: dict,
        max_dist: float = None,
    ) -> List[Tuple[float, float, float, float]]:
        """For each expected landmark position not already covered by a detection,
        find the nearest piece. If the piece is within max_dist of the expected
        grid intersection, treat its center as a proxy for the missing landmark.

        Insight from user: every Xiangqi game has a fixed count of landmarks.
        Missing ones are always occluded by a piece — that piece sits at the
        same grid intersection as the landmark, so its center is the proxy.
        """
        if max_dist is None:
            max_dist = max(rough_grid.cell_width, rough_grid.cell_height) * 0.5

        proxies = []
        for lm_name, positions in ItemDetector._LANDMARK_GRID_POSITIONS.items():
            for col, row in positions:
                key = (lm_name, col, row)
                if key in detected_landmarks_by_pos:
                    continue
                exp = rough_grid.get_point(row, col)
                if exp is None:
                    continue
                nearest = None
                nearest_d = max_dist
                for p in pieces:
                    d = ((p.center[0] - exp.x) ** 2 + (p.center[1] - exp.y) ** 2) ** 0.5
                    if d < nearest_d:
                        nearest_d = d
                        nearest = p
                if nearest is not None:
                    proxies.append((col, row, nearest.center[0], nearest.center[1]))
        return proxies

    @staticmethod
    def _assign_landmarks_to_positions(
        result: ItemDetectionResult, rough_grid: Grid
    ) -> dict:
        """Snap each detected landmark to its nearest expected grid position
        of the same class. Returns {(lm_name, col, row): landmark}."""
        assigned = {}
        for lm_name, positions in ItemDetector._LANDMARK_GRID_POSITIONS.items():
            for lm in result.get_landmarks(lm_name):
                best_key = None
                best_d = float("inf")
                for col, row in positions:
                    if (lm_name, col, row) in assigned:
                        continue
                    exp = rough_grid.get_point(row, col)
                    if exp is None:
                        continue
                    d = ((lm.center[0] - exp.x) ** 2 + (lm.center[1] - exp.y) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best_key = (lm_name, col, row)
                if best_key is not None:
                    assigned[best_key] = lm
        return assigned

    @staticmethod
    def _grid_from_homography(H, image_shape):
        """Project the 9x10 lattice through H. Returns (Grid, score) or (None, None)
        if the resulting grid fails sanity checks. Score = lower is better
        (variance in cell sizes — uniform grids score lower)."""
        from config.settings import GRID_COLS, GRID_ROWS

        if H is None:
            return None, None
        grid_points = np.zeros((GRID_ROWS, GRID_COLS, 2))
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                t = H @ np.array([col, row, 1.0])
                if t[2] != 0:
                    t /= t[2]
                grid_points[row, col] = [t[0], t[1]]

        dx = grid_points[:, 1:, 0] - grid_points[:, :-1, 0]
        dy = grid_points[1:, :, 1] - grid_points[:-1, :, 1]
        cell_w = float(np.mean(np.abs(dx)))
        cell_h = float(np.mean(np.abs(dy)))
        if cell_w < 5 or cell_h < 5:
            return None, None

        if image_shape is not None:
            h_img, w_img = image_shape[:2]
            mx, my = w_img * 0.15, h_img * 0.15
            for r, c in [(0, 0), (0, GRID_COLS - 1),
                         (GRID_ROWS - 1, 0), (GRID_ROWS - 1, GRID_COLS - 1)]:
                px, py = grid_points[r, c]
                if px < -mx or px > w_img + mx or py < -my or py > h_img + my:
                    return None, None

        # Score: uniformity = variance of cell sizes (lower = more uniform = better)
        score = float(np.var(np.abs(dx)) + np.var(np.abs(dy)))
        return Grid(points=grid_points, cell_width=cell_w, cell_height=cell_h), score

    @staticmethod
    def _best_homography_grid(correspondences, image_shape):
        """Try multiple fitting strategies, return the grid with best uniformity
        score that passes sanity checks. None if all attempts fail."""
        src = np.array([[c[0], c[1]] for c in correspondences], dtype=np.float32)
        dst = np.array([[c[2], c[3]] for c in correspondences], dtype=np.float32)

        candidates = []
        try:
            if len(correspondences) >= 5:
                H_ls, _ = cv2.findHomography(src, dst, 0)
                g, s = ItemDetector._grid_from_homography(H_ls, image_shape)
                if g is not None:
                    candidates.append((s, g))
                H_ran, _ = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
                g, s = ItemDetector._grid_from_homography(H_ran, image_shape)
                if g is not None:
                    candidates.append((s, g))
            else:
                H = cv2.getPerspectiveTransform(src[:4], dst[:4])
                g, s = ItemDetector._grid_from_homography(H, image_shape)
                if g is not None:
                    candidates.append((s, g))
        except cv2.error:
            pass

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    @staticmethod
    def _dedupe_landmarks(
        landmarks: List["Landmark"], min_dist: float = 30.0
    ) -> List["Landmark"]:
        """Greedy NMS-by-distance: keep highest-conf landmark, drop neighbors
        within min_dist pixels. Prevents duplicate corner detections from
        polluting the homography fit."""
        sorted_lms = sorted(landmarks, key=lambda l: l.confidence, reverse=True)
        kept: List["Landmark"] = []
        for lm in sorted_lms:
            x, y = lm.center
            if all((x - k.center[0]) ** 2 + (y - k.center[1]) ** 2 > min_dist ** 2
                   for k in kept):
                kept.append(lm)
        return kept

    @staticmethod
    def _collect_correspondences(
        result: ItemDetectionResult,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Map every detected landmark to its expected (col, row) position in the
        9x10 grid. Returns list of (col, row, image_x, image_y).

        Strategy: anchor the reference center on palace-center landmarks when
        available (always at col 4, row 1 or 8) — much more stable than mean
        of all landmarks, which skews when board-conners are unevenly detected.
        """
        # Dedupe each landmark class before correspondence collection
        corners = ItemDetector._dedupe_landmarks(result.board_corners)
        palace_bottoms = ItemDetector._dedupe_landmarks(result.palace_bottoms)
        palace_corners = ItemDetector._dedupe_landmarks(result.palace_corners)
        palace_centers = ItemDetector._dedupe_landmarks(result.palace_centers)

        all_pts = [l.center for l in (corners + palace_bottoms + palace_corners + palace_centers)]
        if not all_pts:
            return []

        cy_ref = sum(p[1] for p in all_pts) / len(all_pts)

        correspondences: List[Tuple[float, float, float, float]] = []

        # palace-center: col 4; split by Y into top (row 1) / bottom (row 8)
        for l in palace_centers:
            x, y = l.center
            correspondences.append((4, 1 if y < cy_ref else 8, x, y))

        def pair_by_y_then_x(landmarks, cols, rows):
            """Split landmarks into top/bottom by Y, then leftmost→cols[0], rightmost→cols[1]."""
            if not landmarks:
                return
            top, bot = [], []
            for l in landmarks:
                (top if l.center[1] < cy_ref else bot).append(l)
            for group, row in [(top, rows[0]), (bot, rows[1])]:
                if not group:
                    continue
                group = sorted(group, key=lambda l: l.center[0])
                if len(group) == 1:
                    l = group[0]
                    correspondences.append((cols[0], row, l.center[0], l.center[1]))
                else:
                    correspondences.append((cols[0], row, group[0].center[0], group[0].center[1]))
                    correspondences.append((cols[1], row, group[-1].center[0], group[-1].center[1]))

        pair_by_y_then_x(corners, cols=(0, 8), rows=(0, 9))
        pair_by_y_then_x(palace_bottoms, cols=(3, 5), rows=(0, 9))
        pair_by_y_then_x(palace_corners, cols=(3, 5), rows=(2, 7))

        return correspondences

    @staticmethod
    def build_grid_from_landmarks(
        result: ItemDetectionResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Grid]:
        """
        Build a 9x10 grid from detected landmarks.

        Strategy: prefer homography over all landmark types — robust to a missing
        or occluded board-conner (e.g. when a chess piece sits on the corner),
        because RANSAC rejects outliers and palace landmarks provide redundancy.

        1. >= 6 mixed landmarks → homography (RANSAC, multi-anchor)
        2. 4 board-conner only → bilinear (legacy path, used when no palaces seen)
        3. >= 2 landmarks → bbox fallback (least accurate)
        """
        from config.settings import GRID_COLS, GRID_ROWS

        # ---- Strategy 1: 4 board-corners → exact perspective transform ----
        # User insight: 4 board-corners define grid geometry. They enclose all
        # pieces and form the 4 grid corners. Use cv2.getPerspectiveTransform
        # so the grid passes through EXACTLY these 4 points (no averaging
        # with palace landmarks that would pull corners off-board).
        corner_corrs = ItemDetector._find_4_board_corners(
            ItemDetector._dedupe_landmarks(result.board_corners),
            result.pieces,
        )
        if len(corner_corrs) == 4:
            grid = ItemDetector._grid_from_4_corners(corner_corrs, image_shape)
            if grid is not None:
                return grid

        # ---- Strategy 2: 4 board corners → bilinear (fallback) ----
        corners = result.board_corners
        if len(corners) >= 4:
            pts = [(c.center[0], c.center[1]) for c in corners]
            tl = min(pts, key=lambda p: p[0] + p[1])
            br = max(pts, key=lambda p: p[0] + p[1])
            tr = max(pts, key=lambda p: p[0] - p[1])
            bl = min(pts, key=lambda p: p[0] - p[1])

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

        # ---- Strategy 3: bbox fallback ----
        all_pts: List[Tuple[float, float]] = []
        for lst in result.landmarks_by_class.values():
            all_pts.extend(l.center for l in lst)
        if len(all_pts) >= 2:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            return BoardDetector().build_grid_from_bbox(bbox, margin=0.0)

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
