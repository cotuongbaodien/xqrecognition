# Backend Integration — Xiangqi Recognition

Tài liệu này dành cho backend team để integrate nhận diện bàn cờ → FEN vào codebase Python hiện có.

## Tổng quan

Đây là Python module: input ảnh (numpy array hoặc file path) → output FEN string.

- 1 model YOLO (`items.pt`) detect 19 classes (14 quân + 5 landmark types)
  - v3-v5: 18 classes (no board-border) — backward compat
  - v6+: 19 classes (added board-border for robust board outline)
- Pipeline: detect → dựng grid 9x10 từ board corners + perimeter → map quân vào ô → FEN
- Không cần REST API, không cần spawn server riêng — gọi như Python function

## Cài đặt

### Bước 1: Copy folder `boarddetection/` vào backend project

Toàn bộ code + model nằm trong 1 folder `boarddetection/`. Copy nguyên folder
vào backend project.

```
your_backend/
├── boarddetection/             ← Copy y nguyên folder này từ repo
│   ├── __init__.py
│   ├── pipeline.py
│   ├── item_detector.py
│   ├── piece_detector.py
│   ├── board_detector.py
│   ├── fen_generator.py
│   ├── rules_validator.py
│   ├── settings.py
│   ├── models/
│   │   └── items.pt           ← ~22MB, copy thủ công (gitignored)
│   └── docs/
│       └── INTEGRATION.md     ← File này
└── your_app.py                ← Code backend của bạn
```

**KHÔNG cần** copy YOLO/ultralytics — install qua pip ở bước 2.

### Bước 2: Install dependencies

```bash
pip install ultralytics opencv-python numpy
```

Hoặc dùng `requirements.txt`:
```
ultralytics>=8.3.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
```

(`torch` tự cài qua `ultralytics`. Nếu có GPU + CUDA, install `torch` với CUDA build.)

## Sử dụng

### Cách dùng cơ bản

```python
from boarddetection import XiangqiRecognizer

# Khởi tạo 1 lần (load model ~5s, sau đó tái dùng)
recognizer = XiangqiRecognizer()  # auto-locates boarddetection/models/items.pt

# Detect từ file path
result = recognizer.recognize("path/to/board.jpg")
print(result.fen)
# → "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
```

### Detect từ numpy array (đã có ảnh trong memory)

```python
import cv2
from boarddetection import XiangqiRecognizer

recognizer = XiangqiRecognizer()  # auto-locates boarddetection/models/items.pt

image = cv2.imread("board.jpg")  # hoặc từ bytes/PIL/etc
result = recognizer.recognize_image(image)
print(result.fen)
```

### Lấy chi tiết từng quân cờ

```python
result = recognizer.recognize("board.jpg")

print(f"FEN: {result.fen}")
print(f"Pieces detected: {len(result.pieces)}")
print(f"Avg confidence: {result.confidence:.2%}")

for piece in result.pieces:
    print(f"  {piece.display_name}: {piece.fen_symbol} "
          f"@ ({piece.center[0]:.0f}, {piece.center[1]:.0f}) "
          f"conf={piece.confidence:.2f}")
```

### Visualization (debug)

```python
import cv2

result = recognizer.recognize("board.jpg", visualize=True)

# result.visualization là numpy array có overlay grid + bbox + landmarks
cv2.imwrite("debug.png", result.visualization)
```

### Pattern Singleton cho production

Tạo **1 instance** dùng chung cả app — tránh load model lại mỗi request:

```python
# xiangqi_service.py
from boarddetection import XiangqiRecognizer

_recognizer = None

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = XiangqiRecognizer()  # auto-locates boarddetection/models/items.pt
    return _recognizer

def detect_fen(image_or_path) -> str:
    """Convenience wrapper."""
    rec = get_recognizer()
    if isinstance(image_or_path, str):
        result = rec.recognize(image_or_path)
    else:
        result = rec.recognize_image(image_or_path)
    return result.fen
```

Trong code backend:
```python
from xiangqi_service import detect_fen

fen = detect_fen("uploaded_board.jpg")
```

## API Reference

### Class `XiangqiRecognizer`

```python
XiangqiRecognizer(items_model_path: str = None)
```

**Parameters:**
- `items_model_path`: đường dẫn `items.pt`. Default = `models/items.pt` relative to project root.

**Methods:**

#### `.recognize(image_path, piece_confidence=0.5, visualize=False) -> RecognitionResult`

Detect từ file ảnh.

#### `.recognize_image(image, piece_confidence=0.5, visualize=False) -> RecognitionResult`

Detect từ numpy array `(H, W, 3) BGR`.

**Tip:** giảm `piece_confidence` xuống `0.3` nếu detect bị miss quân.

### Class `RecognitionResult`

| Attribute | Type | Mô tả |
|---|---|---|
| `fen` | str | FEN string. |
| `pieces` | List[DetectedPiece] | Danh sách quân detect được. |
| `board_state` | BoardState | Bàn cờ 10x9, có thể query từng ô. |
| `confidence` | float | Avg confidence của tất cả quân (0-1). |
| `errors` | List[str] | Cảnh báo nếu có. |
| `visualization` | np.ndarray hoặc None | Ảnh debug nếu `visualize=True`. |

### Class `DetectedPiece`

| Attribute | Type | Mô tả |
|---|---|---|
| `class_name` | str | `"red-chariot"`, `"black-cannon"`, ... (kebab-case) |
| `display_name` | str | `"Xe đỏ"`, `"Pháo đen"`, ... (tiếng Việt) |
| `fen_symbol` | str | `R/N/B/A/K/C/P` (đỏ) hoặc `r/n/b/a/k/c/p` (đen) |
| `confidence` | float | 0-1 |
| `bbox` | (x1, y1, x2, y2) | Bounding box trong ảnh |
| `center` | (cx, cy) | Tâm bbox |

## Format FEN

FEN cờ tướng (Xiangqi):
- **10 hàng** cách nhau `/`, từ hàng 0 (đen, top) đến hàng 9 (đỏ, bottom)
- **Mỗi hàng** có 9 cột (a-i từ trái sang phải)
- **Số** = ô trống liên tiếp (vd `5` = 5 ô trống)
- **Chữ hoa** = quân đỏ, **chữ thường** = quân đen

| Quân | Đỏ | Đen |
|---|---|---|
| Xe (Chariot) | `R` | `r` |
| Mã (Horse) | `N` | `n` |
| Tượng (Elephant) | `B` | `b` |
| Sĩ (Advisor) | `A` | `a` |
| Tướng (General) | `K` | `k` |
| Pháo (Cannon) | `C` | `c` |
| Tốt (Soldier) | `P` | `p` |

**Vị trí khởi đầu chuẩn:**
```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
```

## Lưu ý

### Yêu cầu ảnh đầu vào

| Tiêu chí | Khuyến nghị |
|---|---|
| Resolution | ≥ 640x640 |
| Bàn cờ visibility | Toàn bộ bàn cờ trong khung |
| Bàn cờ chiếm | ≥ 50% diện tích |
| Góc chụp | Thẳng đứng hoặc nghiêng nhẹ < 15° |
| **Orientation** | **Portrait** (board cao hơn rộng) — landscape sẽ bị FEN sai |

### Limitations hiện tại

1. **Mirror trái-phải**: bàn cờ đối xứng → đôi lúc FEN bị mirror. App consuming có thể tự xử lý nếu cần.
2. **Board rotation 90°**: ảnh chụp board nằm ngang → FEN bị rotate. Bắt buộc chụp portrait.
3. **Cold start**: model load ~0.3s (cached) hoặc vài giây (lần đầu, download ultralytics deps). Dùng pattern singleton để chỉ load 1 lần.

### Performance

Measured trên test/5.png (968×610 ảnh thật):

| Hardware | Latency/ảnh | Throughput |
|---|---|---|
| GPU (RTX 3060) | **~17-20ms** | ~50 req/s/worker |
| CPU only | ~500ms-2s | 0.5-2 req/s/worker |

→ Backend không lo overload: 1 worker GPU xử lý ~200K req/giờ.

### Recommended error handling

```python
result = recognizer.recognize_image(image)

if not result.fen or result.fen == "9/9/9/9/9/9/9/9/9/9":
    raise ValueError("Không detect được bàn cờ — yêu cầu user chụp lại")

if result.confidence < 0.5:
    # Cảnh báo nhưng vẫn return FEN
    log.warning(f"Low confidence detection: {result.confidence:.2%}")

if len(result.pieces) < 5:
    raise ValueError("Quá ít quân detect được — ảnh có thể không phải bàn cờ")

if result.errors:
    log.warning(f"Detection warnings: {result.errors}")

return result.fen
```

## Repo

- Repo: https://github.com/cotuongbaodien/xqrecognition
- Branch khuyến nghị: `refactor/single-items-model` (hoặc `main` sau khi merge)
- File model `items.pt` **không có trong git** (gitignored ~22MB) — copy thủ công từ dev machine

## Tùy chọn: chạy như HTTP server

Nếu backend KHÔNG dùng Python (mà Node/Go/Java), có thể chạy `app.py` (FastAPI) như standalone server:

```bash
python app.py --host 0.0.0.0 --port 8000
```

Backend gọi `POST /detect` với `multipart/form-data` (field `file`). Xem `app.py` source để biết endpoints.

Nhưng với Python backend → integrate trực tiếp như trên là tốt nhất (bớt 1 service, bớt overhead network).
