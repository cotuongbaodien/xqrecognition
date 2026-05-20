# Grid Construction Algorithm

Tài liệu mô tả thuật toán dựng grid 9×10 từ ảnh bàn cờ — phần lõi của board detection pipeline. Hiểu rõ thuật toán này giúp debug và improve khi có data mới.

## Problem

Cho ảnh bàn cờ tướng từ user (chụp tự do, có thể xoay/nghiêng/perspective), cần dựng **lưới 9 cột × 10 hàng** trong không gian ảnh để map từng quân cờ vào ô cờ tương ứng.

**Challenge:**
- Bàn cờ có thể bị che (quân cờ ngồi trên góc bàn cờ)
- Camera có thể nghiêng (perspective distortion)
- Ảnh có thể xoay 90°/180° (landscape vs portrait)
- Model detection có false positives/negatives

## Key Insights (from user)

### 1. 34 perimeter grid intersections

Tổng cộng **34 điểm giao trên rìa bàn cờ**:

| Loại landmark | Số điểm | Vị trí trên grid |
|---|---|---|
| `board-conner` | 4 | (0,0), (8,0), (0,9), (8,9) |
| `palace-bottom` | 4 | (3,0), (5,0), (3,9), (5,9) |
| `board-border` | 26 | Còn lại trên 4 cạnh |
| **Tổng** | **34** | Toàn bộ perimeter |

board-border breakdown:
- Top edge (row 0): cols 1, 2, 4, 6, 7 → 5 điểm
- Bottom edge (row 9): cols 1, 2, 4, 6, 7 → 5 điểm
- Left edge (col 0): rows 1-8 → 8 điểm
- Right edge (col 8): rows 1-8 → 8 điểm

### 2. Semantic constraint
- `board-conner` **AT corners** (nơi 4 cạnh giao)
- `board-border` **ON edges, NEVER at corners**
- `palace-bottom` **ON back-rank edge**

### 3. Robustness
Mid-game thường có **10-15 perimeter points bị che**, **vẫn còn 19-24 visible** — đủ để fit quadrilateral chính xác.

### 4. Tận dụng tất cả

> "Phải tận dụng tất cả board-border, board-conner, palace-bottom để fill thành cái khung của grid, bắt buộc tất cả phải nằm ở rìa của grid."

Đây là insight gốc dẫn đến RANSAC line fitting.

## Algorithm Pipeline

```
Detect items (YOLO) → 4 board corners → 9×10 perspective transform grid
                          ↑
                ┌─────────┴─────────┐
                │                   │
        Pieces + perimeter      Detected
        landmarks (anchors)    board-conners (validate)
```

### Stage 1: Find 4 corner candidates

`ItemDetector._find_4_board_corners()` combines all evidence:

```python
candidates = pieces + board_corners + palace_bottoms + board_borders
```

Why include pieces? In **starting position**, chariots sit AT (0,0), (8,0), (0,9), (8,9) — they ARE the corners. Including pieces gives anchors even when board-conner missing.

### Stage 2: Primary method — RANSAC edge lines

`_find_4_edge_lines()` runs **iterative RANSAC**:

1. Find best line through candidate points (max inliers within 8px)
2. Remove inliers
3. Repeat 4 times → 4 dominant lines

Each line should correspond to one board edge.

### Stage 3: Convex hull fallback (if RANSAC fails)

If RANSAC doesn't find 4 distinct lines (only 3 found):
1. `cv2.convexHull` on candidates
2. `cv2.approxPolyDP` simplify to 4 vertices

### Stage 4: 4-extreme fallback (last resort)

If both above fail:
```
TL = candidate with min(x + y)
TR = candidate with max(x - y)
BL = candidate with min(x - y)
BR = candidate with max(x + y)
```

### Stage 5: Universal post-snap (bipartite matching)

Sau khi có 4 corners (từ bất kỳ method nào), nếu có detected `board-conner`:

```python
for each detected board_conner:
    rank computed corners by distance
    snap to closest available corner if <40px
    mark that corner "claimed"
```

**Bipartite matching đảm bảo:** 2 board-conners khác nhau không cùng snap vào 1 computed corner → tránh degenerate grid.

### Stage 6: Piece-color orientation

`_collect_correspondences()` xác định **row axis direction**:

1. **Primary**: line giữa 2 palace-centers (luôn ở col 4, row 1 và 8)
2. **Fallback**: bbox aspect ratio (width > height → landscape, row axis = image-x)
3. **Sign**: project red-centroid vs black-centroid lên row axis. Red phải ở positive direction (row 9). Nếu negative → flip axis.

Algorithm handles 0°, 90° CW, 90° CCW, 180° rotations.

### Stage 7: Perspective transform → 9×10 grid

`cv2.getPerspectiveTransform` với 4 corners → homography matrix H.

Project lưới 9×10 idealized lattice qua H → grid intersections trong image space.

## Validation: Coverage metric

`_verify_grid_with_borders()` đánh giá grid quality:

```
coverage = (# perimeter landmarks within 30% cell-size of any edge) / total
```

- **Coverage 100%**: grid perfect — tất cả detected perimeter landmarks ở trên 4 cạnh
- **Coverage 80-99%**: grid mostly correct, vài detection có noise
- **Coverage <80%**: grid likely misaligned

Adaptive threshold (30% cell size) hợp lý hơn cứng 12px vì scale với image size.

**Note**: coverage là metric chẩn đoán, KHÔNG dùng auto-fallback (vì model false positives làm EM-style fallback overfit).

## Why each piece of the algorithm matters

| Algorithm step | Without it... |
|---|---|
| board-border class | Mid-game corners occluded → grid wrong |
| RANSAC line fit | Single false-positive corner pulls grid skewed (test/5 case) |
| Bipartite snap | Detected corners ignored, grid floats off (commit 8cfc0ba) |
| Piece-color orientation | Landscape boards (test/3 rotated 90°) FEN scrambled |
| Perspective transform | Camera-tilted boards have wrong piece-cell mapping |

## Limitations (known)

1. **Heavy rotation (~30°+)** — test/11. RANSAC finds 4 edges nhưng (col, row) assignment by (x±y) extremes is wrong cho rotated board. Need: orientation detection robust to large rotations (e.g., palace-conner line direction).

2. **Piece misclassifications** — không phải grid issue. Model classifies b↔c, r↔p incorrectly at certain positions. Needs more training data (esp. palace area pieces).

3. **Sparse perimeter (~<6 detections)** — algorithm relies on having enough perimeter points. With very few detections, falls back to convex hull / 4-extreme which is less accurate.

## Files

- `boarddetection/item_detector.py::_find_4_board_corners()` — main entry
- `boarddetection/item_detector.py::_ransac_line()` — single RANSAC iteration
- `boarddetection/item_detector.py::_find_4_edge_lines()` — iterative RANSAC
- `boarddetection/item_detector.py::_fit_corners_from_edge_lines()` — line intersection corner extraction
- `boarddetection/item_detector.py::_verify_grid_with_borders()` — coverage metric
- `boarddetection/item_detector.py::_collect_correspondences()` — piece-color orientation
- `boarddetection/item_detector.py::_grid_from_4_corners()` — perspective transform to 9×10

## Test results (v6 model, current state)

| Test | Coverage | Grid | FEN |
|---|---|---|---|
| 1.png | 100% | ✅ | ✅ EXACT |
| 2.png | 100% | ✅ | ⚠️ piece misclass |
| 3.png | 100% | ✅ | ✅ EXACT (rotated 90°) |
| 4.png | 100% | ✅ | ⚠️ piece misclass |
| 5.png | 89% | ✅ | ⚠️ piece misclass |
| 6.png | 100% | ✅ | ✅ EXACT |
| 7.png | 100% | ✅ | ✅ EXACT |
| 8.png | 100% | ✅ | ✅ EXACT |
| 10.jpg | 83% | ✅ | ⚠️ piece misclass |
| 11.jpg | 94% | ❌ | ❌ rotation ~30° |
| 12.jpg | 100% | ✅ | ⚠️ piece misclass |

**Grid accuracy: 10/11 = 91%**
**FEN exact: 5/11 (remainders are piece classification, not grid)**

## What didn't work (avoid these paths)

1. **Exhaustive line enumeration** (O(N²) pairs): picks spurious diagonals when corners shared between edges.
2. **EM iterative quad fit with auto-fallback**: overfits model false-positive detections. Coverage goes up but grid goes wrong.
3. **3-line synthesis** (when RANSAC finds <4): wrong 4th line position → degenerate grid.
4. **Piece-only corner candidates** (no perimeter): pieces inside board shrink corners inward in mid-game.
5. **Naïve snap** (no bipartite): 2 board-conners collapse to same computed corner.

Anchor commit: `032d42c` on branch `refactor/single-items-model`.
