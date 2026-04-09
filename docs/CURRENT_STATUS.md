# Trạng Thái Hiện Tại - Xiangqi Recognition

**Cập nhật:** 2026-04-09

## Kết quả

**Test trên 52 ảnh fendata (mirror-tolerant metric):**

| Metric | Số ảnh | Tỷ lệ |
|--------|--------|-------|
| Exact FEN match | 29 | 55.8% |
| Mirror match (sau khi flip ngang) | 7 | 13.5% |
| **Total OK (mirror-tolerant)** | **36** | **69.2%** |
| ≥95% cells đúng | 46 | 88.5% |
| ≥90% cells đúng | 50 | 96.2% |
| **Avg cell accuracy** | - | **96.4%** |

**Lý do dùng mirror-tolerant metric:** App cờ tướng của user có chức năng mirror trái-phải, nên FEN bị mirror vẫn dùng được. Chỉ cần thế cờ đúng (vertical orientation đúng + piece positions đúng tương đối).

## Models hiện có

| Model | Mục đích | mAP50 | Backup |
|-------|----------|-------|--------|
| `models/pieces_det.pt` | Detect 14 piece classes (legacy) | 0.99 | `backups/pieces_v1_67pct.pt` |
| `models/items.pt` | Detect 18 classes (pieces + landmarks) | 0.926 | `backups/items_v1_mAP0.926.pt` |
| `models/board_det.pt` | Detect board bounding box | - | - |
| `models/landmarks.pt` | Detect 4 landmark types (legacy) | 0.959 | - |

## Pipeline Hiện Tại

```
Input image
   ↓
1. Pieces detection (pieces_det.pt - legacy 14 classes)
   ↓ 32 pieces max + NMS IoU=0.35
2. Item detection (items.pt - 18 classes)
   ↓ landmarks: board-conner, palace-bottom, palace-center, palace-conner
3. Build grid:
   - Primary: board_det.pt bbox + 2% margin (most stable)
   - Fallback 1: items.pt landmarks → bilinear/homography
   - Fallback 2: board_seg.pt intersections
4. Map pieces to grid (confidence-weighted, distance threshold)
5. Validate game rules (count limits, position constraints)
6. Vertical orientation only (red king at bottom, black king at top)
   - NO horizontal mirror (consuming app handles it)
7. Generate FEN
```

## Class Naming Convention (CHUẨN MỚI)

Theo dataset `itemdetection.yolov8`:
- 14 pieces (kebab-case): `black-advisor`, `black-cannon`, `black-chariot`, `black-elephant`, `black-general`, `black-horse`, `black-soldier`, `red-advisor`, `red-cannon`, `red-chariot`, `red-elephant`, `red-general`, `red-horse`, `red-soldier`
- 4 landmarks: `board-conner`, `palace-bottom`, `palace-center`, `palace-conner`

Xem `config/settings.py` - `ITEM_CLASSES` (mới) và `PIECE_CLASSES` (legacy backward compat).

## Datasets

| Dataset | Số ảnh | Mục đích | Loại label |
|---------|--------|----------|-----------|
| `data/items/` | 153 (122/22/9) | Train items.pt | Manual (chính xác) |
| `data/pieces_merged/` | 1431 train | Train pieces_det.pt | Roboflow original |
| `data/fendata/` | 99 (52 có FEN) | **Evaluation** | FEN ground truth |
| `data/prepare/` | 786 (366 có FEN) | Future evaluation/training | FEN ground truth |

## Vấn đề còn tồn tại

### 1. Mirror trái-phải (~13% ảnh)
- **Bản chất**: Bàn cờ Xiangqi đối xứng trái-phải hoàn toàn
- **Không thể auto-detect** từ vị trí quân (cung tướng đối xứng quanh col 4)
- **Giải pháp**: User app handle mirror → đã chấp nhận, không fix nữa

### 2. Piece detection miss (~30% ảnh)
- Một số ảnh model thiếu 1-2 quân (thường là Rook đã di chuyển)
- Cell accuracy 92-96% nhưng FEN không exact match
- **Giải pháp**: Cần thêm training data (user sẽ cung cấp)

### 3. Image 001 - hoàn toàn fail
- Model không detect được quân nào
- Có thể image quá khác biệt với training data

## Cách Train Model Mới

### Train items.pt (pieces + landmarks)
```bash
python scripts/train_items.py \
    --data data/items/data.yaml \
    --pretrained yolov8s.pt \
    --epochs 200 \
    --batch-size 16 \
    --device cuda \
    --seed 42 \
    --name items_v2
```

Model sẽ được lưu vào `models/items.pt` và backup vào `models/items_items_v2.pt`.

### Train pieces_det.pt (legacy 14 classes)
```bash
python scripts/train_pieces.py \
    --data data/pieces_merged/data.yaml \
    --pretrained yolov8s.pt \
    --epochs 150 \
    --batch-size 16 \
    --device cuda
```

## Cách Đánh Giá

### Mirror-tolerant evaluation (chuẩn hiện tại)
```bash
python scripts/evaluate_fendata.py
# Hoặc quiet mode (không in failed images):
python scripts/evaluate_fendata.py --quiet
```

## Kế Hoạch Tiếp Theo

1. **User cung cấp thêm data** tương tự itemdetection (label thủ công, có landmarks)
2. **Re-train items.pt** với dataset lớn hơn → mAP cao hơn
3. **Mục tiêu**: 80%+ exact match, 90%+ mirror-tolerant
4. **Optional**: Thêm param `side=red|black` vào API để loại bỏ hoàn toàn mirror issue
