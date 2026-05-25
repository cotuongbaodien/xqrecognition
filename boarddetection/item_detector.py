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

from .settings import (
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

    @property
    def board_borders(self) -> List[Landmark]:
        """26 grid-perimeter points (v6+). Combined with board_corners and
        palace_bottoms, gives full 34-point board outline."""
        return self.get_landmarks("board-border")


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
    def _ransac_line(points, threshold=8.0, iterations=200):
        """Single-iteration RANSAC: best line through points. Returns
        (line_vec (vx,vy,x0,y0), inliers, outliers) or (None, [], points)."""
        import random
        if len(points) < 2:
            return None, [], list(points)
        best_inliers_idx = []
        best_pair = None
        for _ in range(iterations):
            i, j = random.sample(range(len(points)), 2)
            p1, p2 = points[i], points[j]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            L = (dx * dx + dy * dy) ** 0.5
            if L < 1e-6:
                continue
            nx, ny = -dy / L, dx / L
            inliers_idx = []
            for k, p in enumerate(points):
                d = abs(nx * (p[0] - p1[0]) + ny * (p[1] - p1[1]))
                if d < threshold:
                    inliers_idx.append(k)
            if len(inliers_idx) > len(best_inliers_idx):
                best_inliers_idx = inliers_idx
                best_pair = (p1, p2)
        if best_pair is None or len(best_inliers_idx) < 2:
            return None, [], list(points)
        inliers = [points[i] for i in best_inliers_idx]
        outliers = [p for k, p in enumerate(points) if k not in set(best_inliers_idx)]
        # Refit precise line via least squares on inliers
        arr = np.array(inliers, dtype=np.float32)
        line = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01)
        return line, inliers, outliers

    @staticmethod
    def _find_4_edge_lines(border_points, threshold=8.0):
        """Iterative RANSAC: find up to 4 dominant lines through border
        points. Each line represents one board edge. Returns list of
        (line, inliers) sorted by inlier count descending."""
        remaining = list(border_points)
        lines = []
        for _ in range(4):
            if len(remaining) < 3:
                break
            line, inliers, outliers = ItemDetector._ransac_line(remaining, threshold)
            if line is None or len(inliers) < 2:
                break
            lines.append((line, inliers))
            remaining = outliers
        return lines

    @staticmethod
    def _verify_grid_with_borders(grid, perimeter_landmarks, threshold=None):
        """Coverage = fraction of detected perimeter landmarks (board-conner
        + palace-bottom + board-border) within `threshold` pixels of any of
        the 4 grid edges. If threshold is None, uses 30% of cell size
        (~½ cell — generous tolerance for sloppy detections).
        Returns coverage in [0, 1].
        """
        if not perimeter_landmarks or grid is None:
            return 1.0
        from .settings import GRID_COLS, GRID_ROWS
        if threshold is None:
            threshold = 0.3 * max(grid.cell_width, grid.cell_height)
        tl = grid.points[0, 0]
        tr = grid.points[0, GRID_COLS - 1]
        bl = grid.points[GRID_ROWS - 1, 0]
        br = grid.points[GRID_ROWS - 1, GRID_COLS - 1]
        edges = [(tl, tr), (tr, br), (br, bl), (bl, tl)]

        def seg_dist(p, a, b):
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 < 1e-6:
                return ((p[0] - ax) ** 2 + (p[1] - ay) ** 2) ** 0.5
            t = max(0.0, min(1.0,
                             ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
            qx, qy = ax + t * dx, ay + t * dy
            return ((p[0] - qx) ** 2 + (p[1] - qy) ** 2) ** 0.5

        on_edge = 0
        for lm in perimeter_landmarks:
            x, y = lm.center
            min_d = min(seg_dist((x, y), a, b) for a, b in edges)
            if min_d < threshold:
                on_edge += 1
        return on_edge / len(perimeter_landmarks)

    @staticmethod
    def _line_intersect(l1, l2):
        v1x, v1y, x01, y01 = l1.flatten()
        v2x, v2y, x02, y02 = l2.flatten()
        A = np.array([[v1x, -v2x], [v1y, -v2y]], dtype=np.float64)
        b = np.array([x02 - x01, y02 - y01], dtype=np.float64)
        try:
            t = np.linalg.solve(A, b)
            return (float(x01 + t[0] * v1x), float(y01 + t[0] * v1y))
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _fit_corners_from_edge_lines(border_points, detected_corners=None,
                                       threshold=10.0):
        """User-suggested approach: find 4 dominant lines through border
        points, intersect adjacent lines → 4 corners. Snap to detected
        board-conner if close (validates and refines).

        Returns (tl, tr, bl, br) or None.
        """
        import math
        if len(border_points) < 6:
            return None

        # Stage 1: Find 4 lines via iterative RANSAC
        lines_with_inliers = ItemDetector._find_4_edge_lines(border_points, threshold)
        if len(lines_with_inliers) < 4:
            return None
        lines = [l for l, _ in lines_with_inliers]

        # Stage 2: group 4 lines into 2 parallel pairs by angle
        angles = []
        for line in lines:
            vx, vy = float(line[0][0]), float(line[1][0])
            a = math.atan2(vy, vx) % math.pi  # [0, π)
            angles.append((a, line))
        angles.sort(key=lambda x: x[0])

        # Try both pairings, pick the one with smaller intra-pair angle diff
        d12_34 = (abs(angles[0][0] - angles[1][0])
                  + abs(angles[2][0] - angles[3][0]))
        d13_24 = (abs(angles[0][0] - angles[2][0])
                  + abs(angles[1][0] - angles[3][0]))
        if d12_34 <= d13_24:
            pair_A = (angles[0][1], angles[1][1])
            pair_B = (angles[2][1], angles[3][1])
        else:
            pair_A = (angles[0][1], angles[2][1])
            pair_B = (angles[1][1], angles[3][1])

        # Stage 3: intersect pair_A × pair_B → 4 quad corners
        quad = []
        for la in pair_A:
            for lb in pair_B:
                pt = ItemDetector._line_intersect(la, lb)
                if pt is None:
                    return None
                quad.append(pt)

        # Stage 4: classify quad corners as TL/TR/BL/BR by (x±y)
        tl = min(quad, key=lambda p: p[0] + p[1])
        br = max(quad, key=lambda p: p[0] + p[1])
        tr = max(quad, key=lambda p: p[0] - p[1])
        bl = min(quad, key=lambda p: p[0] - p[1])

        # Stage 5: snap to detected board-conner if a corner is within
        # snap_threshold pixels of one — validates and refines using model
        # detections that are AT corners by definition.
        if detected_corners:
            bc_pts = [l.center for l in detected_corners]
            snap_threshold = threshold * 4  # generous, e.g. 40px
            def snap(corner):
                if not bc_pts:
                    return corner
                nearest = min(bc_pts,
                              key=lambda p: (p[0] - corner[0]) ** 2 + (p[1] - corner[1]) ** 2)
                d = ((nearest[0] - corner[0]) ** 2
                     + (nearest[1] - corner[1]) ** 2) ** 0.5
                return nearest if d < snap_threshold else corner
            tl = snap(tl)
            tr = snap(tr)
            bl = snap(bl)
            br = snap(br)

        return tl, tr, bl, br

    @staticmethod
    def _fit_corners_from_minarearect(perimeter):
        """Robust fallback for tilted boards: rotated bounding rectangle of
        perimeter landmarks (board-conner, board-border, palace-bottom).
        Handles arbitrary board rotation since minAreaRect is rotation-aware.
        Returns (tl, tr, bl, br) ordered in image space, or None.
        """
        if len(perimeter) < 4:
            return None
        pts = np.array(perimeter, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        corners = [(float(p[0]), float(p[1])) for p in box]
        tl = min(corners, key=lambda p: p[0] + p[1])
        br = max(corners, key=lambda p: p[0] + p[1])
        tr = max(corners, key=lambda p: p[0] - p[1])
        bl = min(corners, key=lambda p: p[0] - p[1])
        if len({tl, tr, bl, br}) != 4:
            return None
        return tl, tr, bl, br

    @staticmethod
    def _refine_corners_with_borders(rough_corners, border_points):
        """Given a rough quadrilateral (tl, tr, bl, br) and detected border
        landmarks, assign each border to its nearest of 4 edges, fit a line
        to each edge's points, and intersect adjacent lines for refined
        corners. Edges with <2 borders keep the rough edge.

        Critical when the rough quad comes from 3 corners + parallelogram
        completion or minAreaRect — both can be slightly off the actual
        board edges, so borders pull the edges to their true positions.
        """
        if not border_points:
            return rough_corners
        r_tl, r_tr, r_bl, r_br = rough_corners
        rough_edges = {
            "top":    (r_tl, r_tr),
            "right":  (r_tr, r_br),
            "bottom": (r_br, r_bl),
            "left":   (r_bl, r_tl),
        }

        def seg_dist(p, a, b):
            ax, ay = a
            bx, by = b
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 < 1e-6:
                return ((p[0] - ax) ** 2 + (p[1] - ay) ** 2) ** 0.5
            t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
            qx, qy = ax + t * dx, ay + t * dy
            return ((p[0] - qx) ** 2 + (p[1] - qy) ** 2) ** 0.5

        xs = [p[0] for p in border_points] + [c[0] for c in rough_corners]
        ys = [p[1] for p in border_points] + [c[1] for c in rough_corners]
        diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        threshold = diag * 0.10  # accept borders within 10% of diag from rough edge

        edge_pts = {k: [] for k in rough_edges}
        for pt in border_points:
            best, best_d = None, float("inf")
            for name, (a, b) in rough_edges.items():
                d = seg_dist(pt, a, b)
                if d < best_d:
                    best_d = d
                    best = name
            if best_d <= threshold:
                edge_pts[best].append(pt)

        edge_lines = {}
        for name, points in edge_pts.items():
            if len(points) >= 2:
                arr = np.array(points, dtype=np.float32)
                edge_lines[name] = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01)
            else:
                a, b = rough_edges[name]
                dx, dy = b[0] - a[0], b[1] - a[1]
                L = (dx * dx + dy * dy) ** 0.5 or 1.0
                edge_lines[name] = np.array(
                    [[dx / L], [dy / L], [a[0]], [a[1]]], dtype=np.float32
                )

        def line_intersect(l1, l2):
            v1x, v1y, x01, y01 = l1.flatten()
            v2x, v2y, x02, y02 = l2.flatten()
            A = np.array([[v1x, -v2x], [v1y, -v2y]], dtype=np.float64)
            b = np.array([x02 - x01, y02 - y01], dtype=np.float64)
            try:
                t = np.linalg.solve(A, b)
                return (float(x01 + t[0] * v1x), float(y01 + t[0] * v1y))
            except np.linalg.LinAlgError:
                return None

        tl = line_intersect(edge_lines["top"], edge_lines["left"])
        tr = line_intersect(edge_lines["top"], edge_lines["right"])
        bl = line_intersect(edge_lines["bottom"], edge_lines["left"])
        br = line_intersect(edge_lines["bottom"], edge_lines["right"])
        if None in (tl, tr, bl, br):
            return rough_corners
        return tl, tr, bl, br

    @staticmethod
    def _fit_corners_from_perimeter(perimeter):
        """All perimeter points lie on 4 board edges. Fit 4 edge lines and
        compute the 4 corners as line intersections — corner positions are
        NEVER a single input point, but the geometric intersection of edges.

        Steps:
          1. Rough quadrilateral via convex hull + approxPolyDP (initial corners)
          2. Assign each perimeter point to its nearest of 4 edges
          3. cv2.fitLine on each edge's points → 4 precise edge lines
          4. Pairwise intersections (top∩left, top∩right, bottom∩left, bottom∩right)
        """
        pts = np.array(perimeter, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        peri_len = cv2.arcLength(hull, True)
        approx = None
        for eps_frac in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
            a = cv2.approxPolyDP(hull, eps_frac * peri_len, True)
            if len(a) == 4:
                approx = a
                break
        if approx is None:
            return None

        rough = [(float(p[0][0]), float(p[0][1])) for p in approx]
        r_tl = min(rough, key=lambda p: p[0] + p[1])
        r_br = max(rough, key=lambda p: p[0] + p[1])
        r_tr = max(rough, key=lambda p: p[0] - p[1])
        r_bl = min(rough, key=lambda p: p[0] - p[1])
        rough_edges = {
            "top":    (r_tl, r_tr),
            "right":  (r_tr, r_br),
            "bottom": (r_br, r_bl),
            "left":   (r_bl, r_tl),
        }

        def seg_dist(p, a, b):
            ax, ay = a
            bx, by = b
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 < 1e-6:
                return ((p[0] - ax) ** 2 + (p[1] - ay) ** 2) ** 0.5
            t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
            qx, qy = ax + t * dx, ay + t * dy
            return ((p[0] - qx) ** 2 + (p[1] - qy) ** 2) ** 0.5

        # Distance threshold: 8% of bounding-box diagonal. Tight enough to
        # reject interior pieces (>1 cell deep), loose enough to catch
        # corner pieces and perimeter landmarks slightly off the rough edge.
        xs = [p[0] for p in perimeter]
        ys = [p[1] for p in perimeter]
        diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        threshold = diag * 0.08

        edge_pts = {k: [] for k in rough_edges}
        for pt in perimeter:
            best, best_d = None, float("inf")
            for name, (a, b) in rough_edges.items():
                d = seg_dist(pt, a, b)
                if d < best_d:
                    best_d = d
                    best = name
            if best_d <= threshold:
                edge_pts[best].append(pt)

        # Need at least 2 points per edge to fit a line
        edge_lines = {}
        for name, points in edge_pts.items():
            if len(points) >= 2:
                arr = np.array(points, dtype=np.float32)
                edge_lines[name] = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01)
            else:
                # Use the rough edge as the line
                a, b = rough_edges[name]
                dx, dy = b[0] - a[0], b[1] - a[1]
                L = (dx * dx + dy * dy) ** 0.5 or 1.0
                edge_lines[name] = np.array(
                    [[dx / L], [dy / L], [a[0]], [a[1]]], dtype=np.float32
                )

        def line_intersect(l1, l2):
            v1x, v1y, x01, y01 = l1.flatten()
            v2x, v2y, x02, y02 = l2.flatten()
            A = np.array([[v1x, -v2x], [v1y, -v2y]], dtype=np.float64)
            b = np.array([x02 - x01, y02 - y01], dtype=np.float64)
            try:
                t = np.linalg.solve(A, b)
                return (float(x01 + t[0] * v1x), float(y01 + t[0] * v1y))
            except np.linalg.LinAlgError:
                return None

        tl = line_intersect(edge_lines["top"], edge_lines["left"])
        tr = line_intersect(edge_lines["top"], edge_lines["right"])
        bl = line_intersect(edge_lines["bottom"], edge_lines["left"])
        br = line_intersect(edge_lines["bottom"], edge_lines["right"])
        if None in (tl, tr, bl, br):
            return None
        return tl, tr, bl, br

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
        palace_centers: Optional[List["Landmark"]] = None,
        palace_bottoms: Optional[List["Landmark"]] = None,
        board_borders: Optional[List["Landmark"]] = None,
    ) -> List[Tuple[float, float, float, float]]:
        """Find the 4 board-corners AND assign each to its board (col, row).

        Priority for corner candidates (most reliable first):

        1. **Perimeter landmarks** (board-conner + palace-bottom + board-border)
           if we have ≥8 points. All 34 perimeter grid intersections lie on
           the board boundary, so the 4 extremes of perimeter detections
           directly give the 4 board corners. Robust to occlusion (we expect
           15-30 of 34 to be visible in any game state).
        2. **Pieces + corners** fallback when perimeter is sparse. The 4
           extreme pieces are at or near the board corners.

        Step 2 — assign (col, row) using piece-color orientation and palace
        landmarks. Standard FEN: red at row 9, black at row 0. Vector from
        black-centroid to red-centroid is the row axis.
        """
        candidates = [p.center for p in pieces]
        if detected_corners:
            candidates.extend([l.center for l in detected_corners])
        if palace_bottoms:
            candidates.extend([l.center for l in palace_bottoms])
        if board_borders:
            candidates.extend([l.center for l in board_borders])
        if not candidates:
            return []

        tl = tr = bl = br = None
        n_corners = len(detected_corners) if detected_corners else 0
        refine_pts = []
        if board_borders:
            refine_pts.extend([l.center for l in board_borders])
        if palace_bottoms:
            refine_pts.extend([l.center for l in palace_bottoms])

        # =========================================================
        # STRATEGY BY DETECTED-CORNER COUNT
        # =========================================================
        # The 4 board corners are the strongest grid anchors. Strategy
        # branches by how many are detected, falling back to perimeter
        # geometry when corners are sparse.

        # --- 4 corners: use directly (no line-fitting needed) ---
        if n_corners >= 4:
            bc_pts = [l.center for l in detected_corners]
            cand_tl = min(bc_pts, key=lambda p: p[0] + p[1])
            cand_br = max(bc_pts, key=lambda p: p[0] + p[1])
            cand_tr = max(bc_pts, key=lambda p: p[0] - p[1])
            cand_bl = min(bc_pts, key=lambda p: p[0] - p[1])
            if len({cand_tl, cand_tr, cand_bl, cand_br}) == 4:
                tl, tr, bl, br = cand_tl, cand_tr, cand_bl, cand_br

        # --- 3 corners: parallelogram completion + edge refinement ---
        # D = A + C - B where A,C are diagonal endpoints (max pairwise
        # distance) and B is the shared corner. Refinement pulls each
        # edge to the line through nearest borders.
        elif n_corners == 3:
            bc_pts = [l.center for l in detected_corners]
            c1, c2, c3 = bc_pts
            d12 = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
            d23 = (c2[0] - c3[0]) ** 2 + (c2[1] - c3[1]) ** 2
            d13 = (c1[0] - c3[0]) ** 2 + (c1[1] - c3[1]) ** 2
            if d12 >= d23 and d12 >= d13:
                c4 = (c1[0] + c2[0] - c3[0], c1[1] + c2[1] - c3[1])
            elif d23 >= d13:
                c4 = (c2[0] + c3[0] - c1[0], c2[1] + c3[1] - c1[1])
            else:
                c4 = (c1[0] + c3[0] - c2[0], c1[1] + c3[1] - c2[1])
            all_4 = [c1, c2, c3, c4]
            cand_tl = min(all_4, key=lambda p: p[0] + p[1])
            cand_br = max(all_4, key=lambda p: p[0] + p[1])
            cand_tr = max(all_4, key=lambda p: p[0] - p[1])
            cand_bl = min(all_4, key=lambda p: p[0] - p[1])
            if len({cand_tl, cand_tr, cand_bl, cand_br}) == 4:
                rough = (cand_tl, cand_tr, cand_bl, cand_br)
                refined = ItemDetector._refine_corners_with_borders(
                    rough, refine_pts
                )
                tl, tr, bl, br = refined

        # --- 2 corners: minAreaRect anchored on perimeter+corners ---
        # The 2 detected corners are guaranteed to be 2 board corners.
        # Including them in minAreaRect input ensures the rect's hull
        # touches them. SNAP step (later) pulls the 2 nearest rect
        # corners exactly onto the detected ones.
        # --- 1 corner: same approach. The 1 corner anchors snap step.
        # --- 0 corners: pure perimeter geometry.
        # All three cases use the same minAreaRect + edge-refine path,
        # differentiated only by SNAP at the end.

        def quadrilateral_sane(c_tl, c_tr, c_bl, c_br):
            """Reject quadrilaterals where any corner is wildly outside the
            candidate point cloud — happens when RANSAC line-fit picks
            near-parallel lines whose intersection diverges."""
            xs = [p[0] for p in candidates]
            ys = [p[1] for p in candidates]
            mx, Mx = min(xs), max(xs)
            my, My = min(ys), max(ys)
            margin = max(Mx - mx, My - my) * 0.3
            lo_x, hi_x = mx - margin, Mx + margin
            lo_y, hi_y = my - margin, My + margin
            for p in (c_tl, c_tr, c_bl, c_br):
                if not (lo_x <= p[0] <= hi_x and lo_y <= p[1] <= hi_y):
                    return False
            return True

        def quad_non_degenerate(c_tl, c_tr, c_bl, c_br):
            """Reject quads where corners are collinear or width/height
            degenerate. A real board has 4 corners forming a proper quad with
            min-side-length > 20% of bbox diag."""
            sides = [
                ((c_tl[0] - c_tr[0]) ** 2 + (c_tl[1] - c_tr[1]) ** 2) ** 0.5,
                ((c_tr[0] - c_br[0]) ** 2 + (c_tr[1] - c_br[1]) ** 2) ** 0.5,
                ((c_br[0] - c_bl[0]) ** 2 + (c_br[1] - c_bl[1]) ** 2) ** 0.5,
                ((c_bl[0] - c_tl[0]) ** 2 + (c_bl[1] - c_tl[1]) ** 2) ** 0.5,
            ]
            xs = [c[0] for c in (c_tl, c_tr, c_bl, c_br)]
            ys = [c[1] for c in (c_tl, c_tr, c_bl, c_br)]
            diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
            return min(sides) >= diag * 0.2

        # Common path for 0/1/2 corners: minAreaRect of perimeter+pieces,
        # refined by per-edge line fit. Robust to arbitrary rotation and
        # mild perspective. minAreaRect is deterministic. Pieces anchor
        # against false-positive perimeter detections. The SNAP step
        # later pulls computed corners to detected corners (if any).
        if tl is None and len(candidates) >= 4:
            corners = ItemDetector._fit_corners_from_minarearect(candidates)
            if (corners is not None
                    and quadrilateral_sane(*corners)
                    and quad_non_degenerate(*corners)):
                refine_with = list(refine_pts)
                if detected_corners:
                    refine_with.extend([l.center for l in detected_corners])
                corners = ItemDetector._refine_corners_with_borders(
                    corners, refine_with
                )
                tl, tr, bl, br = corners

        # Fallback 1: RANSAC edge-line fit (axis-aligned tight boards where
        # borders form clear lines). Less robust to rotation but more precise
        # when 4 edges have many borders evenly distributed.
        if tl is None and board_borders and len(board_borders) >= 6:
            border_pts = [l.center for l in board_borders]
            if palace_bottoms:
                border_pts.extend([l.center for l in palace_bottoms])
            if detected_corners:
                border_pts.extend([l.center for l in detected_corners])
            corners = ItemDetector._fit_corners_from_edge_lines(
                border_pts, detected_corners=detected_corners
            )
            if (corners is not None
                    and quadrilateral_sane(*corners)
                    and quad_non_degenerate(*corners)):
                tl, tr, bl, br = corners

        # Fallback 2: rough-quad + per-edge line-fit (axis-aligned boards)
        if tl is None:
            corners = ItemDetector._fit_corners_from_perimeter(candidates)
            if (corners is not None
                    and quadrilateral_sane(*corners)
                    and quad_non_degenerate(*corners)):
                tl, tr, bl, br = corners

        # Fallback 3: 4-extreme by (x±y)
        if tl is None:
            tl = min(candidates, key=lambda p: p[0] + p[1])
            br = max(candidates, key=lambda p: p[0] + p[1])
            tr = max(candidates, key=lambda p: p[0] - p[1])
            bl = min(candidates, key=lambda p: p[0] - p[1])

        # Universal post-snap with quadrant matching: each detected
        # board-conner maps to the computed corner it's closest to, but no
        # two board-conners can map to the same computed corner (greedy
        # bipartite matching). board-conner detections are AT corners so
        # they refine line-fit output without collapsing distinct corners.
        if detected_corners:
            bc_pts = [l.center for l in detected_corners]
            slots = {"TL": tl, "TR": tr, "BL": bl, "BR": br}
            taken = set()
            SNAP_THRESHOLD = 40.0
            for bc in bc_pts:
                ranked = sorted(
                    slots.items(),
                    key=lambda kv: (bc[0] - kv[1][0]) ** 2
                                   + (bc[1] - kv[1][1]) ** 2,
                )
                for slot_name, slot_pt in ranked:
                    if slot_name in taken:
                        continue
                    d = ((bc[0] - slot_pt[0]) ** 2
                         + (bc[1] - slot_pt[1]) ** 2) ** 0.5
                    if d < SNAP_THRESHOLD:
                        slots[slot_name] = bc
                        taken.add(slot_name)
                    break
            tl, tr, bl, br = slots["TL"], slots["TR"], slots["BL"], slots["BR"]

        extremes = [tl, tr, bl, br]

        def standard_portrait():
            return [
                (0, 0, tl[0], tl[1]),
                (8, 0, tr[0], tr[1]),
                (0, 9, bl[0], bl[1]),
                (8, 9, br[0], br[1]),
            ]

        # Step 2: derive row axis DIRECTION
        # Priority order:
        # 1. Line between 2 palace-centers (geometric, palace-centers at col 4
        #    rows 1 and 8, so line IS the row axis)
        # 2. 4-corner bounding box aspect ratio. A Xiangqi board has 10 rows
        #    × 9 cols, so the row axis is the LONGER dimension of the board.
        #    Width > height → board is landscape → row axis is horizontal.
        # 3. Default image-y (assume portrait)
        row_axis = np.array([0.0, 1.0])  # image-y default
        if palace_centers and len(palace_centers) >= 2:
            p1 = np.array(palace_centers[0].center, dtype=float)
            p2 = np.array(palace_centers[1].center, dtype=float)
            v = p2 - p1
            if np.linalg.norm(v) > 100:
                row_axis = v / np.linalg.norm(v)
        else:
            xs = [p[0] for p in extremes]
            ys = [p[1] for p in extremes]
            bbox_w = max(xs) - min(xs)
            bbox_h = max(ys) - min(ys)
            # Board has 10 rows × 9 cols → row axis aligns with the LONGER side.
            # In image space:
            #   - landscape (w > h): row axis is horizontal (image-x)
            #   - portrait  (h > w): row axis is vertical (image-y)
            if bbox_w > bbox_h:
                row_axis = np.array([1.0, 0.0])

        # Step 3: use piece colors to determine SIGN (which end is row 9)
        red_pieces = [p for p in pieces if p.fen_symbol and p.fen_symbol.isupper()]
        black_pieces = [p for p in pieces if p.fen_symbol and p.fen_symbol.islower()]

        # If both colors detected with meaningful separation along row_axis,
        # flip row_axis to point toward red side.
        if len(red_pieces) >= 2 and len(black_pieces) >= 2:
            red_c = np.array([
                sum(p.center[0] for p in red_pieces) / len(red_pieces),
                sum(p.center[1] for p in red_pieces) / len(red_pieces),
            ])
            black_c = np.array([
                sum(p.center[0] for p in black_pieces) / len(black_pieces),
                sum(p.center[1] for p in black_pieces) / len(black_pieces),
            ])
            sep = np.dot(red_c - black_c, row_axis)
            if abs(sep) < 30:  # too small along this axis → ambiguous, keep default
                pass
            elif sep < 0:  # red is on the negative side → flip
                row_axis = -row_axis

        col_axis = np.array([row_axis[1], -row_axis[0]])

        cx = sum(p[0] for p in extremes) / 4
        cy = sum(p[1] for p in extremes) / 4

        correspondences = []
        for x, y in extremes:
            v = np.array([x - cx, y - cy])
            row_proj = float(np.dot(v, row_axis))
            col_proj = float(np.dot(v, col_axis))
            row = 9 if row_proj > 0 else 0
            col = 8 if col_proj > 0 else 0
            correspondences.append((col, row, x, y))

        seen = {(c[0], c[1]) for c in correspondences}
        if len(seen) != 4:
            return standard_portrait()
        return correspondences

    @staticmethod
    def _corners_to_correspondences(
        tl, tr, bl, br,
        pieces: List[DetectedPiece],
        palace_centers: Optional[List["Landmark"]] = None,
        palace_corners: Optional[List["Landmark"]] = None,
        palace_bottoms: Optional[List["Landmark"]] = None,
    ) -> List[Tuple[float, float, float, float]]:
        """Assign 4 image-space corners (already TL/TR/BL/BR by x±y) to grid
        (col,row).

        ROW AXIS (which way rows run) is determined from palace landmarks,
        which are the most reliable orientation cue: both palaces lie on the
        center columns (3-5) at the two row-ends, so palace points spread
        primarily along the row axis. Priority:
          1. 2 palace-centers (cols 4, rows 1 & 8) → line between them
          2. PCA major axis of ALL palace points (center+conner+bottom) when
             they span >3 cells (i.e. points from BOTH palaces present)
          3. 4-corner bbox aspect ratio (board's long side = row axis)

        SIGN (which end is row 9 = red) from piece colors.
        """
        extremes = [tl, tr, bl, br]

        def standard_portrait():
            return [
                (0, 0, tl[0], tl[1]), (8, 0, tr[0], tr[1]),
                (0, 9, bl[0], bl[1]), (8, 9, br[0], br[1]),
            ]

        xs = [p[0] for p in extremes]
        ys = [p[1] for p in extremes]
        board_diag = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5

        palace_pts = []
        for grp in (palace_centers, palace_corners, palace_bottoms):
            if grp:
                palace_pts.extend([l.center for l in grp])

        row_axis = None
        # 1. Two palace-centers → cleanest row axis
        if palace_centers and len(palace_centers) >= 2:
            p1 = np.array(palace_centers[0].center, dtype=float)
            p2 = np.array(palace_centers[1].center, dtype=float)
            v = p2 - p1
            if np.linalg.norm(v) > 0.3 * board_diag:
                row_axis = v / np.linalg.norm(v)
        # 2. PCA major axis of all palace points (robust to partial detection)
        if row_axis is None and len(palace_pts) >= 2:
            pts = np.array(palace_pts, dtype=float)
            mean = pts.mean(axis=0)
            centered = pts - mean
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            major = vt[0]
            # spread along major axis must indicate two palaces (>30% diag)
            proj = centered @ major
            if (proj.max() - proj.min()) > 0.3 * board_diag:
                row_axis = major / np.linalg.norm(major)
        # 3. bbox aspect fallback
        if row_axis is None:
            row_axis = np.array([0.0, 1.0])
            if (max(xs) - min(xs)) > (max(ys) - min(ys)):
                row_axis = np.array([1.0, 0.0])

        # SIGN: red side = row 9. Use piece-color centroids projected on row axis.
        red = [p for p in pieces if p.fen_symbol and p.fen_symbol.isupper()]
        black = [p for p in pieces if p.fen_symbol and p.fen_symbol.islower()]
        if len(red) >= 2 and len(black) >= 2:
            red_c = np.array([sum(p.center[0] for p in red) / len(red),
                              sum(p.center[1] for p in red) / len(red)])
            black_c = np.array([sum(p.center[0] for p in black) / len(black),
                                sum(p.center[1] for p in black) / len(black)])
            sep = np.dot(red_c - black_c, row_axis)
            if abs(sep) >= 30 and sep < 0:
                row_axis = -row_axis

        col_axis = np.array([row_axis[1], -row_axis[0]])
        cx = sum(p[0] for p in extremes) / 4
        cy = sum(p[1] for p in extremes) / 4
        corrs = []
        for x, y in extremes:
            v = np.array([x - cx, y - cy])
            row = 9 if float(np.dot(v, row_axis)) > 0 else 0
            col = 8 if float(np.dot(v, col_axis)) > 0 else 0
            corrs.append((col, row, x, y))
        if len({(c[0], c[1]) for c in corrs}) != 4:
            return standard_portrait()
        return corrs

    @staticmethod
    def build_grid_from_quad(
        quad: Tuple[Tuple[float, float], ...],
        pieces: List[DetectedPiece],
        palace_centers: Optional[List["Landmark"]] = None,
        palace_corners: Optional[List["Landmark"]] = None,
        palace_bottoms: Optional[List["Landmark"]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Grid]:
        """Build a 9x10 grid from 4 board corners (from segmentation mask).
        quad is (tl, tr, bl, br) ordered by x±y extremes."""
        tl, tr, bl, br = quad
        corrs = ItemDetector._corners_to_correspondences(
            tl, tr, bl, br, pieces, palace_centers,
            palace_corners, palace_bottoms,
        )
        return ItemDetector._grid_from_4_corners(corrs, image_shape)

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
        from .settings import GRID_COLS, GRID_ROWS
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

        # Euclidean cell size (works for rotated/perspective grids — X/Y
        # component alone collapses to ~0 for a 90°-rotated board)
        dx_vec = grid_points[:, 1:, :] - grid_points[:, :-1, :]
        dy_vec = grid_points[1:, :, :] - grid_points[:-1, :, :]
        cell_w = float(np.mean(np.linalg.norm(dx_vec, axis=2)))
        cell_h = float(np.mean(np.linalg.norm(dy_vec, axis=2)))
        return Grid(points=grid_points, cell_width=cell_w, cell_height=cell_h)

    @staticmethod
    def _build_bilinear_from_corners(
        corner_corrs: List[Tuple[float, float, float, float]],
    ) -> Optional[Grid]:
        """Build a Grid by bilinear interpolation between the 4 corner
        correspondences. Each corr is (col, row, x, y) where col,row ∈
        {(0,0),(8,0),(0,9),(8,9)}."""
        from .settings import GRID_COLS, GRID_ROWS
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
        from .settings import GRID_COLS, GRID_ROWS

        if H is None:
            return None, None
        grid_points = np.zeros((GRID_ROWS, GRID_COLS, 2))
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                t = H @ np.array([col, row, 1.0])
                if t[2] != 0:
                    t /= t[2]
                grid_points[row, col] = [t[0], t[1]]

        # Euclidean cell size (rotation-invariant)
        dx_vec = grid_points[:, 1:, :] - grid_points[:, :-1, :]
        dy_vec = grid_points[1:, :, :] - grid_points[:-1, :, :]
        dx_norm = np.linalg.norm(dx_vec, axis=2)
        dy_norm = np.linalg.norm(dy_vec, axis=2)
        cell_w = float(np.mean(dx_norm))
        cell_h = float(np.mean(dy_norm))
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

        score = float(np.var(dx_norm) + np.var(dy_norm))
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
        from .settings import GRID_COLS, GRID_ROWS

        # ---- Strategy 1: 4 board-corners → exact perspective transform ----
        # User insight: 4 board-corners define grid geometry. They enclose all
        # pieces and form the 4 grid corners. Use cv2.getPerspectiveTransform
        # so the grid passes through EXACTLY these 4 points (no averaging
        # with palace landmarks that would pull corners off-board).
        corner_corrs = ItemDetector._find_4_board_corners(
            ItemDetector._dedupe_landmarks(result.board_corners),
            result.pieces,
            palace_centers=ItemDetector._dedupe_landmarks(result.palace_centers),
            palace_bottoms=ItemDetector._dedupe_landmarks(result.palace_bottoms),
            board_borders=ItemDetector._dedupe_landmarks(result.board_borders),
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
