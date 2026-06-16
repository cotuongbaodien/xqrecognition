"""Diagnostic: on each test image, compare the CURRENT 90deg-canonicalization
(aspect ratio of board quad) against the PALACE-AXIS method (vector between the
two palace centroids = the rows/rank axis). Prints both decisions so we can
confirm the palace axis is a reliable orientation signal before wiring it in.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/backups/board_seg_v5.pt"
IMG_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "data/board_seg_v5/test/images")

BOARD_CLS, PALACE_CLS = 0, 1
model = YOLO(MODEL)


def reduce_to_quad(contour):
    peri = cv2.arcLength(contour, True)
    for eps in (0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2)
    return cv2.boxPoints(cv2.minAreaRect(contour))


def order_quad(pts):
    pts = [(float(p[0]), float(p[1])) for p in pts]
    tl = min(pts, key=lambda p: p[0] + p[1])
    br = max(pts, key=lambda p: p[0] + p[1])
    tr = max(pts, key=lambda p: p[0] - p[1])
    bl = min(pts, key=lambda p: p[0] - p[1])
    return tl, tr, bl, br


imgs = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
print(f"model={MODEL}  images={len(imgs)}\n")
for ip in imgs:
    img = cv2.imread(str(ip))
    r = model(img, conf=0.25, verbose=False)[0]
    if r.masks is None:
        print(f"{ip.name:45s} NO MASKS")
        continue
    cls = r.boxes.cls.cpu().numpy().astype(int)
    polys = r.masks.xy
    board_polys = [p for p, c in zip(polys, cls) if c == BOARD_CLS]
    palace_polys = [p for p, c in zip(polys, cls) if c == PALACE_CLS]

    if not board_polys:
        print(f"{ip.name:45s} no board mask  palaces={len(palace_polys)}")
        continue
    board = max(board_polys, key=lambda p: cv2.contourArea(p.astype(np.float32)))
    tl, tr, bl, br = order_quad(reduce_to_quad(board.astype(np.float32)))

    cols_edge = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    rows_edge = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
    aspect_says_rotate = cols_edge > 1.05 * rows_edge

    # palace centroids
    pcs = [p.astype(np.float32).mean(axis=0) for p in palace_polys]
    palace_axis = None
    if len(pcs) >= 2:
        # use the two largest palaces
        pcs = sorted(palace_polys, key=lambda p: cv2.contourArea(p.astype(np.float32)), reverse=True)[:2]
        c1 = pcs[0].astype(np.float32).mean(axis=0)
        c2 = pcs[1].astype(np.float32).mean(axis=0)
        v = c2 - c1
        # rows axis direction; compare its alignment to board edges
        cols_vec = np.array([tr[0] - tl[0], tr[1] - tl[1]])
        rows_vec = np.array([bl[0] - tl[0], bl[1] - tl[1]])
        align_cols = abs(np.dot(v, cols_vec)) / (np.linalg.norm(v) * np.linalg.norm(cols_vec) + 1e-9)
        align_rows = abs(np.dot(v, rows_vec)) / (np.linalg.norm(v) * np.linalg.norm(rows_vec) + 1e-9)
        # palace axis SHOULD align with rows. If it aligns more with cols -> rotate
        palace_says_rotate = align_cols > align_rows
        palace_axis = f"alignCols={align_cols:.2f} alignRows={align_rows:.2f} -> rotate={palace_says_rotate}"

    agree = ""
    if palace_axis is not None:
        agree = "  AGREE" if (("rotate=True" in palace_axis) == aspect_says_rotate) else "  ***DISAGREE***"
    print(f"{ip.name:45s} palaces={len(palace_polys)} aspect_rotate={aspect_says_rotate}  palace[{palace_axis}]{agree}")
