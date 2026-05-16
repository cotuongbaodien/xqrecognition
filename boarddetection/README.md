# boarddetection

Python package nhận diện bàn cờ tướng (Xiangqi) từ ảnh → FEN notation.

## Quick start

```python
from boarddetection import XiangqiRecognizer

recognizer = XiangqiRecognizer()  # tự load models/items.pt trong package
result = recognizer.recognize("board.jpg")
print(result.fen)  # "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
```

## Cài đặt vào backend

1. Copy nguyên folder `boarddetection/` (bao gồm `models/items.pt` ~22MB) vào project Python của bạn.
2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python numpy
   ```
3. Import và dùng (xem `docs/INTEGRATION.md` cho chi tiết).

**KHÔNG cần** copy YOLO/ultralytics — install qua pip.

## Performance

| Hardware | Latency | Throughput |
|---|---|---|
| GPU (RTX 3060) | ~17-20ms | ~50 req/s/worker |
| CPU only | ~500ms-2s | 0.5-2 req/s/worker |

Model warm-up: load lần đầu ~0.3s, các lần sau cached.

## Cấu trúc

```
boarddetection/
├── __init__.py            ← Public API: XiangqiRecognizer, render_fen_ascii, ...
├── pipeline.py            ← Main entry (XiangqiRecognizer)
├── item_detector.py       ← YOLO wrapper, detect 18 classes (pieces + landmarks)
├── piece_detector.py      ← DetectedPiece dataclass + utilities
├── board_detector.py      ← Grid + visualization helpers
├── fen_generator.py       ← BoardState → FEN string
├── rules_validator.py     ← Xiangqi rule check (1 general/side, etc.)
├── settings.py            ← Class names, FEN mapping, paths, thresholds
├── models/
│   └── items.pt           ← Trained YOLOv8 model (22MB, gitignored)
└── docs/
    └── INTEGRATION.md     ← Hướng dẫn integrate chi tiết cho backend team
```

## Update model

Khi có model mới (vd retrain với data tốt hơn):
1. Copy file `items.pt` mới vào `boarddetection/models/items.pt`
2. Restart service (model load on init)
3. Không cần đụng code.

Khi có code update:
1. Pull branch mới hoặc copy folder `boarddetection/` mới
2. Import path không đổi → backend code không cần sửa.
