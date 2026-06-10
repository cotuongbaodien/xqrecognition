# boarddetection

Python package nhận diện bàn cờ tướng (Xiangqi) từ ảnh → FEN notation.

## Quick start

```python
from boarddetection import XiangqiRecognizer

recognizer = XiangqiRecognizer()  # tự load models/items.pt (+ board_seg.pt nếu có)
result = recognizer.recognize("board.jpg")
print(result.fen)  # "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
```

## Cài đặt vào backend

1. Copy nguyên folder `boarddetection/` (bao gồm `models/items.pt` ~19MB và
   `models/board_seg.pt` ~6MB) vào project Python của bạn.
2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python numpy
   ```
3. Import và dùng (xem `docs/INTEGRATION.md` cho chi tiết).

**KHÔNG cần** copy YOLO/ultralytics — install qua pip.

## Kiến trúc (2 model)

| Model | Loại | Vai trò |
|---|---|---|
| `items.pt` | YOLO detect (v8, **18 class** = 14 quân + 4 landmark) | 1 forward pass ra cả quân cờ lẫn landmark (`board-conner`, `palace-center`, `palace-conner`, `palace-bottom`). Chạy ở **imgsz=960** (quân nhỏ trong ảnh full bàn → recall cao hơn, đỡ nhầm mã/xe). |
| `board_seg.pt` | YOLO-seg | Khoanh vùng bàn cờ → polygon → rút về **4 góc** (tl,tr,bl,br). Robust với ảnh nghiêng/phối cảnh hơn cách fit từ landmark. Đây là model dùng để **vẽ polygon bàn cờ** + dựng lưới. |

`board_seg.pt` là **optional nhưng được ưu tiên**: nếu có thì dùng quad của seg (4 góc được "snap" về landmark `board-conner` cho chính xác từng pixel khi bàn cờ rõ nét). Nếu thiếu file hoặc seg fail → tự fallback sang dựng lưới thuần từ landmark.

### Pipeline (`XiangqiRecognizer.recognize_image`)

1. **items.pt** (imgsz=960) → quân cờ + landmark trong 1 lượt.
2. NMS quân (IoU 0.35), cap tối đa 32 quân.
3. **Dựng lưới**: `board_seg.pt` → quad 4 góc → snap vào `board-conner` → `build_grid_from_quad` (dùng thêm palace landmark). Fallback: `build_grid_from_landmarks`.
4. Map quân vào ô lưới.
5. Chuẩn hoá chiều DỌC (đỏ dưới / đen trên). **Không** lật ngang — app phía tiêu thụ tự xử mirror.
6. Validate + sửa theo luật cờ (`rules_validator`).
7. Sinh FEN.

## Performance

| Hardware | Latency (imgsz=960) | Throughput |
|---|---|---|
| GPU (RTX 3060) | ~28-31ms | ~30-40 req/s/worker |
| CPU only | ~0.8-3s | <1 req/s/worker |

Model warm-up: load lần đầu ~0.3-0.5s, các lần sau cached. (imgsz=960 thêm ~+11ms/frame so với 640 nhưng exact-FEN tăng 48→54 trên bộ test 86 ảnh.)

## Cấu trúc

```
boarddetection/
├── __init__.py            ← Public API: XiangqiRecognizer, render_fen_ascii, ...
├── pipeline.py            ← Main entry (XiangqiRecognizer)
├── item_detector.py       ← items.pt wrapper (18 class), dựng grid từ quad/landmark
├── board_segmenter.py     ← board_seg.pt wrapper (YOLO-seg) → polygon bàn cờ → 4 góc
├── piece_detector.py      ← DetectedPiece dataclass + NMS + visualization
├── board_detector.py      ← Grid + vẽ lưới
├── fen_generator.py       ← BoardState → FEN string
├── rules_validator.py     ← Check luật (1 tướng/bên, v.v.) + tự sửa
├── settings.py            ← Class names, FEN mapping, paths, thresholds
├── server.py              ← FastAPI service (/detect, /health)
├── models/
│   ├── items.pt           ← Detect model (~19MB, gitignored) — v8: 18 class (14 quân + 4 landmark)
│   └── board_seg.pt       ← Seg model (~6MB, gitignored) — khoanh bàn cờ → 4 góc
└── docs/
    ├── INTEGRATION.md     ← Hướng dẫn integrate chi tiết cho backend
    └── GRID_ALGORITHM.md  ← Thuật toán dựng lưới từ quad/landmark
```

## Cập nhật

> ⚠️ **Cập nhật model thì CHỈ copy file `.pt`, ĐỪNG copy nguyên folder.**
> Copy cả folder sẽ đè `README.md` + code `.py` bằng bản cũ và làm hỏng đồng bộ
> (đã từng làm mất `imgsz=960` và mô tả `board_seg`). Model `.pt` được
> gitignore + bind-mount nên copy riêng là đủ.

**Đổi model (thường xuyên):**
1. Copy **chỉ** file `items.pt` (hoặc `board_seg.pt`) mới vào `boarddetection/models/`.
2. Restart service để nạp lại: `docker restart abcengine-ocr`.
3. Không đụng code, không rebuild.

**Đổi code (hiếm):**
1. Sửa/pull code qua **git** (đừng copy đè folder).
2. Rebuild image: `docker compose up -d --build ocr`.
3. Import path không đổi → backend không cần sửa.

**Nguồn code chuẩn (latest):** `xqrecognition/boarddetection` — luôn lấy code từ đây.
Model train xong copy thẳng file `.pt` vào `models/` của nơi deploy.
