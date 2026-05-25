# Xiangqi Recognition System

Nhận diện bàn cờ tướng (Xiangqi) từ ảnh → xuất **FEN notation**.

Input: ảnh chụp bàn cờ (bất kỳ góc, nghiêng, livestream, app online, bàn gỗ thật).
Output: grid 9×10, tọa độ quân, và chuỗi FEN.

```
Ảnh → segment bàn cờ → 4 góc → perspective grid 9×10 → detect quân → snap vào grid → FEN
```

---

## Hướng giải quyết (Approach)

Nguyên tắc cốt lõi: **AI lo nhận diện, toán học lo grid.** Tách rõ 2 việc:

1. **Localize bàn cờ** = AI (segmentation) — robust với mọi góc nghiêng/perspective.
2. **Dựng grid** = hình học thuần (homography từ 4 góc) — không cần AI, luôn chính xác.

### Pipeline chi tiết

| Bước | Cách làm | Module |
|---|---|---|
| 1. Localize bàn | **Segmentation mask** (YOLO11n-seg) → polygon → 4 góc grid | `board_segmenter.py` |
| 2. Tinh chỉnh góc | Snap 4 góc seg vào `board-conner` detected (pixel-precise) | `pipeline._snap_quad_to_corners` |
| 3. Xác định chiều | Row axis từ **palace landmarks** (PCA của palace-center/conner/bottom); chiều đỏ/đen từ màu quân | `item_detector._corners_to_correspondences` |
| 4. Dựng grid 9×10 | `cv2.getPerspectiveTransform` (4 góc → 90 giao điểm) | `item_detector._grid_from_4_corners` |
| 5. Detect quân | YOLOv8s (`items.pt`) trên ảnh GỐC (giữ chất lượng) | `item_detector.py` |
| 6. Snap + FEN | Map quân vào giao điểm gần nhất → ma trận 10×9 → FEN | `fen_generator.py` |

**Fallback:** nếu segmentation fail, tự động chuyển sang dựng grid từ landmark points (board-conner + board-border + palace).

### Tại sao segmentation thay vì detect điểm

Cách cũ detect các điểm rời rạc trên viền (board-conner, board-border) rồi fit polygon — kém robust khi bàn nghiêng (board-conner mAP chỉ ~0.66, hay thiếu điểm). Segmentation nhìn **toàn bộ mặt bàn** nên ra polygon ổn định kể cả khi nghiêng 30° hoặc che góc.

---

## Models

| Model | Kiến trúc | Nhiệm vụ |
|---|---|---|
| `boarddetection/models/items.pt` | YOLOv8s, 19 classes | Detect 14 quân + 5 landmark trong 1 pass |
| `boarddetection/models/board_seg.pt` | YOLO11n-seg, 1 class | Segment polygon bàn cờ |

Backups (model cũ) ở `models/backups/` — `boarddetection/models/` chỉ chứa model production.

### 19 classes của items.pt (kebab-case)

- **14 quân**: `{black,red}-{advisor,cannon,chariot,elephant,general,horse,soldier}`
- **5 landmark**: `board-border`, `board-conner`, `palace-bottom`, `palace-center`, `palace-conner`

> Với segmentation, `board-border` (26 điểm/bàn) hầu như **không còn cần** cho path chính — chỉ dùng ở fallback. Landmark thiết yếu hiện tại: `board-conner` (snap góc) + `palace-*` (xác định chiều).

---

## Usage

### Detection

```bash
# 1 ảnh
python detect.py --image board.jpg --output output/

# cả thư mục
python detect.py --dir test/ --output test/output/ --confidence 0.3
```

### Python API

```python
from boarddetection import XiangqiRecognizer

recognizer = XiangqiRecognizer()          # tự load items.pt + board_seg.pt
result = recognizer.recognize("board.jpg")
print(result.fen)                          # → "1rbakabnr/9/1cn3c2/..."
```

### Training

```bash
# Train detection (pieces + landmarks)
python scripts/train_items.py --data data/items_vN/data.yaml --name items_vN

# Train board segmentation
yolo segment train data=data/board_seg/data.yaml model=yolo11n-seg.pt epochs=150 imgsz=640

# Pipeline retrain tự động (extract → split → backup → train → test)
python scripts/retrain.py --zip data/itemdetection.yolov8.zip --name items_vN
```

---

## Trạng thái hiện tại (2026-05-25)

Test trên 14 ảnh đa dạng (web, ảnh chụp thật, app, livestream):

- ✅ **Grid đúng 14/14** — kể cả bàn nghiêng (11), perspective (13), top-down (15)
- **FEN EXACT: 7/14**
- Lỗi còn lại **100% là piece classification** (nhầm loại/màu quân), KHÔNG phải grid
  - Ví dụ 10.jpg (chụp qua màn hình): màu đỏ bị ám → nhầm đen↔đỏ
  - Giải pháp: train pieces với data đa dạng hơn

### Hướng cải thiện tiếp

1. **Re-label + train pieces** với nhiều style (screen-capture, ánh sáng khác) → fix color/type misclass
2. Seg train thêm ảnh đa dạng → corner precision cao hơn
3. (optional) Adaptive HSV color clustering per-image nếu cần fix màu không qua training

---

## Project Structure

```
xqrecognition/
├── boarddetection/              # Package chính (self-contained, deploy được)
│   ├── pipeline.py              # XiangqiRecognizer — entry point
│   ├── board_segmenter.py       # Segment bàn → 4 góc
│   ├── item_detector.py         # Detect quân+landmark, dựng grid
│   ├── piece_detector.py        # NMS + visualization
│   ├── fen_generator.py         # Map grid → FEN
│   ├── rules_validator.py       # Validate luật cờ
│   ├── settings.py              # 19 classes, config
│   ├── models/                  # items.pt + board_seg.pt (chỉ production)
│   └── docs/                    # GRID_ALGORITHM.md, INTEGRATION.md
├── scripts/                     # train_items, gen_board_polygon, retrain, ...
├── data/                        # Datasets
├── test/                        # Ảnh test + output visualizations
└── detect.py                    # Detection CLI
```

## Requirements

- Python 3.8+, PyTorch 2.0+, Ultralytics, OpenCV, NumPy
- CUDA (optional, cho GPU)

## License

MIT
