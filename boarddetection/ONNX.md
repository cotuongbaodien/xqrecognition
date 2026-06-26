# ONNX CPU backend — boarddetection

Cho phép pipeline chạy **CPU bằng ONNX Runtime** (không torch/ultralytics) để deploy
lên VPS không-GPU, GIỮ NGUYÊN toàn bộ hậu xử lý/FEN. Bản GPU `.pt` (ultralytics) vẫn
chạy y như cũ — chọn backend bằng 1 biến môi trường.

## Bật/tắt
```bash
OCR_MODEL_FORMAT=onnx   # pipeline nạp items.onnx + board_seg.onnx (CPU, onnxruntime)
OCR_MODEL_FORMAT=pt     # (mặc định) nạp items.pt + board_seg.pt (GPU, ultralytics)
OCR_ONNX_THREADS=4      # (tuỳ chọn) số luồng intra-op của onnxruntime; trống = auto
```
`pipeline.py` đọc env này và đổi đuôi file model. `item_detector`/`board_segmenter`
`load_model()` tự branch: path `.onnx` → `OnnxYOLO`, ngược lại → `ultralytics.YOLO`.
Import `ultralytics` đã được làm **lazy** (chỉ import khi thực sự nạp `.pt`) nên image
CPU không cần torch/ultralytics.

## Thành phần
| File | Vai trò |
|---|---|
| `onnx_backend.py` | `OnnxYOLO(path, task)` — onnxruntime + numpy + cv2. `__call__(img, conf, imgsz)` trả `[result]` mimic interface ultralytics: `result.boxes.cls/.conf/.xyxy` (detect) và `result.masks.xy` + `result.boxes.cls` (segment). |
| `models/items.onnx` | detect 18 lớp, imgsz **960** (export từ `items.pt`). |
| `models/board_seg.onnx` | segment board+palace, imgsz **640** (export từ `board_seg.pt`). |
| `Dockerfile.cpu` | image lean (~1GB): onnxruntime + opencv-headless + numpy, KHÔNG torch. |

## Export lại `.onnx` (khi đổi `.pt`)
Cần ultralytics+torch tạm (dùng image GPU `ocr-gpu:latest`):
```bash
docker run --rm -v $PWD/models:/models ocr-gpu:latest bash -lc \
  "cd /models && yolo export model=items.pt format=onnx imgsz=960 opset=12 \
   && yolo export model=board_seg.pt format=onnx imgsz=640 opset=12"
```
> ⚠️ Mỗi lần retrain `.pt` → PHẢI export lại `.onnx` rồi verify parity (dưới).

## Chi tiết postprocess (khớp ultralytics, để khỏi sai FEN)
- **Preprocess:** letterbox vuông `imgsz×imgsz` giữ tỉ lệ, pad 114, BGR→RGB, `/255`, NCHW float32. Lưu `ratio`+`pad` để un-letterbox về toạ độ gốc.
- **Detect:** output `[1, 4+nc, N]` → transpose → conf = max class prob → lọc `>=conf` → **NMS theo lớp** (agnostic=False, **iou=0.7** = default ultralytics predict) → xywh→xyxy → un-letterbox.
- **Segment:** thêm proto `[1,32,mh,mw]` → `mask = sigmoid(coeff @ proto)` → crop theo box (scale mask) → upsample imgsz → threshold 0.5 → `cv2.findContours` lấy contour lớn nhất/instance → un-letterbox → `result.masks.xy`.

## Verify parity (BẮT BUỘC sau mỗi lần đổi model/code)
Dùng bench 243 ảnh (`xqrecognition/test/bench` + `scripts/eval_bench.py` logic). So
**exact-FEN ONNX vs torch** — không được tụt. Lần build đầu (2026-06-26):

| Backend | exact-FEN | MISS | EXTRA | WRONG | tốc độ |
|---|---|---|---|---|---|
| torch `.pt` (GPU) | 228/243 | 29 | 15 | 39 | 76ms |
| **ONNX (CPU)** | **230/243** | 8 | 0 | 38 | ~210ms |

→ ONNX không tụt (≥ torch). Lệch nhiều → soi NMS iou (0.7), threshold mask (0.5),
letterbox pad (114), conf (`settings.py`: PIECE 0.25 / BOARD seg 0.25).

Deploy VPS: xem `../docs/onnx-cpu-vps-handoff.md` mục 5.
