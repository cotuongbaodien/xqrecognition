"""Board segmentation: locate the Xiangqi board outline via a YOLO-seg model.

Returns the 4 grid corners of the board, plus (with the v5+ 2-class model)
the two palace centroids and a per-corner "clipped at image border" flag.

More robust than landmark-point detection for tilted/perspective boards
because the segmentation model sees the whole board region, not sparse
intersection points.

Class scheme:
  v1-v4 (single class): everything is the board outline.
  v5+   (2 classes): 0 = xiangqi-board, 1 = xiangqi-palace (2 instances/board).
The two palace centroids define the RANK (rows) axis — the line between them
runs along the central file, parallel to the 10-rank direction — which fixes
the 90deg orientation ambiguity that board aspect-ratio alone cannot resolve
(portrait / oblique-perspective captures).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

BOARD_CLASS_ID = 0
PALACE_CLASS_ID = 1
# A corner this close (px) to the image edge means the board is cut off there.
BORDER_CLIP_PX = 3.0


@dataclass
class BoardSegResult:
    """Segmentation output for one board."""
    quad: Tuple[Tuple[float, float], ...]          # (tl, tr, bl, br)
    palace_centers: List[Tuple[float, float]]      # 0-2 palace centroids
    palace_quads: List[Tuple[Tuple[float, float], ...]]  # 0-2 palace 4-corner quads
    clipped: List[bool]                            # per-corner: on image edge?

    @property
    def any_clipped(self) -> bool:
        return any(self.clipped)


class BoardSegmenter:
    """Segments the board (and palaces) and extracts the 4 grid corners."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        if str(model_path).endswith(".onnx"):
            from .onnx_backend import OnnxYOLO
            self.model = OnnxYOLO(model_path, task="segment")
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    def detect(
        self, image: np.ndarray, confidence: float = 0.25, imgsz: int = 640
    ) -> Optional[BoardSegResult]:
        """Run the seg model and return the board quad + palace centroids +
        per-corner clip flags, or None if no board mask is found.

        imgsz is pinned to 640: ultralytics otherwise infers at the model's
        training imgsz (960 for v5), where the board mask empirically localizes
        the outer grid corners WORSE than at 640."""
        if self.model is None:
            return None
        result = self.model(image, conf=confidence, imgsz=imgsz, verbose=False)[0]
        if result.masks is None or len(result.masks) == 0:
            return None

        polys = result.masks.xy
        if not polys:
            return None
        # Split masks by class. The legacy single-class model has only class 0,
        # so palace_polys is simply empty there (graceful degradation).
        if result.boxes is not None and result.boxes.cls is not None:
            cls = result.boxes.cls.cpu().numpy().astype(int)
        else:
            cls = np.zeros(len(polys), dtype=int)
        board_polys = [p for p, c in zip(polys, cls) if c == BOARD_CLASS_ID]
        palace_polys = [p for p, c in zip(polys, cls) if c == PALACE_CLASS_ID]
        if not board_polys:
            return None

        # Largest board mask = the board (ignore spurious small masks)
        biggest = max(
            board_polys, key=lambda p: cv2.contourArea(p.astype(np.float32))
        ).astype(np.float32)
        quad = self._reduce_to_quad(biggest)
        if quad is None:
            return None

        # The two largest palace masks (the model can emit a spurious 3rd;
        # keep the 2 biggest, they are the real palaces). For each we keep both
        # the centroid (orientation cue) and the 4-corner quad (precise
        # interior grid anchors — these ARE the palace-conner/palace-bottom
        # intersection points, cols 3/5 x rows 0/2 or 7/9).
        palace_centers: List[Tuple[float, float]] = []
        palace_quads: List[Tuple[Tuple[float, float], ...]] = []
        for poly in sorted(
            palace_polys,
            key=lambda p: cv2.contourArea(p.astype(np.float32)),
            reverse=True,
        )[:2]:
            pf = poly.astype(np.float32)
            c = pf.mean(axis=0)
            palace_centers.append((float(c[0]), float(c[1])))
            pq = self._reduce_to_quad(pf)
            if pq is not None:
                palace_quads.append(
                    tuple((float(p[0]), float(p[1])) for p in pq))

        ordered = self._order_quad(quad, palace_centers)
        if ordered is None:
            return None

        h, w = image.shape[:2]
        clipped = [self._is_clipped(p, w, h) for p in ordered]
        return BoardSegResult(
            quad=ordered, palace_centers=palace_centers,
            palace_quads=palace_quads, clipped=clipped,
        )

    def get_board_quad(
        self, image: np.ndarray, confidence: float = 0.25
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        """Back-compat: return just (tl, tr, bl, br), or None."""
        res = self.detect(image, confidence)
        return res.quad if res is not None else None

    # ------------------------------------------------------------------ #
    # corner ordering / canonicalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_clipped(pt: Tuple[float, float], w: int, h: int) -> bool:
        x, y = pt
        return (
            x <= BORDER_CLIP_PX or x >= w - 1 - BORDER_CLIP_PX
            or y <= BORDER_CLIP_PX or y >= h - 1 - BORDER_CLIP_PX
        )

    @classmethod
    def _order_quad(
        cls,
        quad: np.ndarray,
        palace_centers: List[Tuple[float, float]],
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        """Label the 4 quad vertices (tl, tr, bl, br) and canonicalize the
        90deg rotation. tl->tr is mapped to the 9 columns, tl->bl to the 10
        rows by the grid builder, so the rows edge must align with the RANK
        axis. With 2 palaces we use the palace->palace vector (definitive);
        otherwise we fall back to the board aspect ratio (10 ranks > 9 files,
        so the longer edge is the rows axis) — fragile under perspective, hence
        only a last resort."""
        pts = [(float(p[0]), float(p[1])) for p in quad]

        tl = min(pts, key=lambda p: p[0] + p[1])
        br = max(pts, key=lambda p: p[0] + p[1])
        tr = max(pts, key=lambda p: p[0] - p[1])
        bl = min(pts, key=lambda p: p[0] - p[1])
        if len({tl, tr, bl, br}) != 4:
            # x±y extremes collapse near a 45deg (diamond) tilt — fall back to
            # angular ordering around the centroid (always 4 distinct labels).
            tl, tr, bl, br = cls._order_quad_angular(pts)

        # --- 90deg canonicalization -------------------------------------
        cols_vec = np.array([tr[0] - tl[0], tr[1] - tl[1]])
        rows_vec = np.array([bl[0] - tl[0], bl[1] - tl[1]])
        cols_edge = float(np.hypot(*cols_vec))
        rows_edge = float(np.hypot(*rows_vec))

        rotate = False
        if len(palace_centers) >= 2:
            # palace axis IS the rank axis: it should align with the rows edge.
            v = np.array(palace_centers[1]) - np.array(palace_centers[0])
            nv = np.linalg.norm(v)
            if nv > 1e-6:
                align_cols = abs(float(np.dot(v, cols_vec))) / (
                    nv * (cols_edge + 1e-9))
                align_rows = abs(float(np.dot(v, rows_vec))) / (
                    nv * (rows_edge + 1e-9))
                # palace axis aligns with cols edge -> cols is really the rank
                # axis -> rotate labels 90deg so it becomes the rows edge.
                rotate = align_cols > align_rows
        else:
            # No reliable palace cue: assume board taller than wide (ratio
            # 1.125), so the rows edge is the longer one.
            rotate = cols_edge > 1.05 * rows_edge

        if rotate:
            tl, tr, bl, br = tr, br, tl, bl
        return tl, tr, bl, br

    @staticmethod
    def _order_quad_angular(
        pts: List[Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], ...]:
        """Tilt-robust corner labeling: sort the 4 points clockwise around
        their centroid, anchor the ring at the corner nearest the top-left
        (min x+y), then read off tl, tr, br, bl in CW order."""
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4
        # Clockwise in image coords (y grows downward): sort by -atan2.
        ring = sorted(pts, key=lambda p: -np.arctan2(p[1] - cy, p[0] - cx))
        start = min(range(4), key=lambda i: ring[i][0] + ring[i][1])
        ring = ring[start:] + ring[:start]
        tl, tr, br, bl = ring[0], ring[1], ring[2], ring[3]
        return tl, tr, bl, br

    @staticmethod
    def _reduce_to_quad(contour: np.ndarray) -> Optional[np.ndarray]:
        """Reduce a polygon contour to exactly 4 vertices, PRESERVING its true
        (possibly trapezoidal) shape under perspective.

        approxPolyDP often steps straight past 4 (e.g. 6 -> 3) when the mask
        edge has small bumps, in which case the old minAreaRect fallback
        returned an axis-aligned rectangle and destroyed the perspective
        foreshortening. Instead we take the coarsest approximation that still
        has >= 4 vertices and drop the least-important vertices (smallest
        triangle area with their neighbours) until exactly 4 remain — this
        keeps the 4 real corners and discards mid-edge bumps."""
        peri = cv2.arcLength(contour, True)
        candidate = None
        for eps_frac in (0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10):
            approx = cv2.approxPolyDP(contour, eps_frac * peri, True).reshape(-1, 2)
            if len(approx) == 4:
                return approx
            if len(approx) > 4:
                candidate = approx  # coarser each step -> fewer, closer to 4
            else:
                break
        if candidate is not None and len(candidate) > 4:
            return BoardSegmenter._drop_to_4(candidate)
        # Last resort: rotated bounding rectangle (loses perspective)
        return cv2.boxPoints(cv2.minAreaRect(contour))

    @staticmethod
    def _drop_to_4(pts: np.ndarray) -> np.ndarray:
        """Iteratively remove the vertex whose triangle with its two neighbours
        has the smallest area, until 4 vertices remain."""
        pts = [tuple(map(float, p)) for p in pts]
        while len(pts) > 4:
            n = len(pts)
            def tri_area(i):
                a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
                return abs((b[0] - a[0]) * (c[1] - a[1])
                           - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
            j = min(range(n), key=tri_area)
            pts.pop(j)
        return np.array(pts, dtype=np.float32)
