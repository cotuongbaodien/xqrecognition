# OCR → VPS (CPU) + ONNX — bàn giao team

> Mục tiêu: dời OCR (board recognition, YOLO11) từ máy-nhà-GPU sang **VPS portal
> (103.175.146.124, 8 core / 16GB / KHÔNG GPU)**, chạy **CPU bằng ONNX Runtime**
> (bỏ torch/ultralytics cho gọn + nhanh hơn torch-CPU).

---

## ✅ TRẠNG THÁI: ĐÃ BUILD + VERIFY XONG (2026-06-26)

Toàn bộ phần kỹ thuật (mục 2–4) **đã làm xong và verify parity**. Code-level doc:
[`boarddetection/ONNX.md`](../boarddetection/ONNX.md). Chỉ còn **mục 5 (deploy VPS)** —
phần này user tự làm (cần SSH VPS + Cloudflare dashboard).

**File đã chuẩn bị (đã port sang `xqrecognition/boarddetection` — nguồn canonical):**
| File | Trạng thái |
|---|---|
| `boarddetection/onnx_backend.py` | ✅ MỚI — `OnnxYOLO` (onnxruntime+numpy, mimic ultralytics) |
| `boarddetection/models/items.onnx` (36MB), `board_seg.onnx` (11MB) | ✅ export opset 12, imgsz 960/640 |
| `boarddetection/Dockerfile.cpu` | ✅ MỚI — image lean ~1GB (không torch) |
| `docker-compose.cpu.yml` (root `ocr-gpu-service/`) | ✅ MỚI — stack CPU, port 8001, `OCR_MODEL_FORMAT=onnx` |
| `item_detector.py` / `board_segmenter.py` | ✅ SỬA — `load_model` branch `.onnx` |
| `board_detector.py` / `piece_detector.py` | ✅ SỬA — import ultralytics lazy |
| `pipeline.py` | ✅ SỬA — chọn backend theo env `OCR_MODEL_FORMAT` (default `pt`) |

**Bật ONNX:** `OCR_MODEL_FORMAT=onnx` (đã set sẵn trong `Dockerfile.cpu` + `docker-compose.cpu.yml`).
Bản GPU `.pt` (default) **không bị ảnh hưởng** — đã verify code sửa chạy `.pt` = baseline.

**Verify parity (bench 243 ảnh, exact-FEN mirror-tolerant):**
| Backend | exact-FEN | MISS | EXTRA | WRONG | tốc độ |
|---|---|---|---|---|---|
| torch `.pt` (GPU, baseline) | 228/243 | 29 | 15 | 39 | 76ms |
| **ONNX (CPU)** | **230/243** | 8 | 0 | 38 | ~210ms (script) / 187ms (HTTP) |
| code-sửa trên `.pt` (regression check) | **228/243** (giống baseline) | — | — | — | 75ms |

→ ONNX **không tụt** (≥ torch), CPU ~0.2s/ảnh (nhanh hơn nhiều mức 0.5–1s dự đoán).
HTTP smoke-test ocr-cpu: `/health` ok, no-auth→401, Flow A→200, Flow B→200, FEN đúng.

> ⚠️ Mỗi lần retrain `.pt` → export lại `.onnx` + chạy lại parity (xem `boarddetection/ONNX.md`).

---

## 0. Quyết định tốc độ (đọc trước khi làm)
- GPU hiện tại: **<50ms/ảnh**. CPU torch: **~1–2s**. **ONNX CPU: ~0.5–1s** (nhanh hơn
  torch-CPU ~1.5–2×, nhưng **không gần GPU**).
- OCR **ít dùng** → ~1s chấp nhận được. Nếu cần real-time/đồng thời cao → **giữ GPU máy nhà**.
- ONNX lợi: bỏ torch (image ~1.5GB → ~400MB), RAM thấp, nhanh hơn torch-CPU. KHÔNG biến CPU thành GPU.

---

## 1. Hiện trạng (để khỏi phá)
- Stack: `ocr-gpu-service/` — FastAPI (`boarddetection/server.py`, port **8001**), endpoint
  `/health` + `/detect`. Auth header `X-OCR-Secret` = `OCR_SHARED_SECRET` (khớp portal).
- **2 model YOLO11** (`boarddetection/models/`, bind-mount, ~25MB):
  - `items.pt` — **detection** 18 lớp (14 quân + 4 landmark). imgsz **960**, conf ~0.25–0.3.
  - `board_seg.pt` — **segmentation** (board + palace). imgsz **640**, conf ~0.25.
- Chạy qua Cloudflare tunnel `ocr-gpu` → hostname **`ocr-gpu.abcxq.app`** (KHÔNG phải `ocr.abcxq.app`,
  README cũ ghi sai). Portal `.env.production`: `OCR_BACKEND_URL` + `OCR_PUBLIC_URL` đều trỏ host này.
- Code dùng `ultralytics.YOLO` → đây là chỗ phải thay bằng ONNX.

---

## 2. ⭐ Interface ultralytics mà code đang đọc (PHẢI giữ y hệt khi viết shim ONNX)
ONNX shim chỉ cần trả về object có đúng các field dưới — **toàn bộ hậu xử lý/FEN giữ nguyên**:

**`item_detector.py` (detection)** — `item_detector.py:106–140`:
```python
results = self.model(image, conf=confidence, imgsz=960, verbose=False)
for r in results:
    r.boxes.cls[i]   # int class id (0..17)
    r.boxes.conf[i]  # float
    r.boxes.xyxy[i]  # (x1,y1,x2,y2) theo TOẠ ĐỘ ẢNH GỐC
```
→ Shim cần: list 1 `result`, `result.boxes` có `.cls/.conf/.xyxy` (index được, toạ độ gốc).
(Code gọi `.cpu().numpy()` — hoặc shim mimic, hoặc sửa 3 dòng đọc numpy thẳng.)

**`board_segmenter.py` (segmentation)** — `board_segmenter.py:67–81`:
```python
result = self.model(image, conf=confidence, imgsz=640, verbose=False)[0]
result.masks            # None nếu không có
result.masks.xy         # list[ ndarray Nx2 ] — polygon mỗi instance, TOẠ ĐỘ GỐC
result.boxes.cls        # ndarray class id (0=board, 1=palace)
```
→ Shim cần: `result.masks.xy` (list polygon) + `result.boxes.cls`. Đây là phần **khó nhất**
(YOLO11-seg: proto-mask + coeff → mask → contour → polygon, phải khớp ultralytics).

> ⚠️ Lưu ý mơ hồ: `settings.py BOARD_SEG_CLASSES={0:"inters"}` nhưng `board_segmenter` coi
> 0=board/1=palace. **Xác nhận `board_seg.pt` đang deploy là model board/palace 2-lớp** (theo
> segmenter), không phải model "inters". (`board_detector.py`, `piece_detector.py` là LEGACY —
> pipeline active chỉ dùng `item_detector` + `board_segmenter`; xem `pipeline.py`.)

---

## 3. Việc ONNX (các bước) — ✅ ĐÃ LÀM XONG
### 3.1 Export .pt → .onnx (một lần, cần ultralytics+torch tạm)
```bash
yolo export model=items.pt     format=onnx imgsz=960 opset=12   # → items.onnx
yolo export model=board_seg.pt format=onnx imgsz=640 opset=12   # → board_seg.onnx (kèm proto)
```
(Chạy trong container `pytorch/pytorch` + `pip install ultralytics onnxruntime`. Giữ .onnx vào `models/`.)

### 3.2 Viết `onnx_backend.py` (onnxruntime + numpy + cv2, KHÔNG torch)
Một class `OnnxYOLO(path, task)` với `__call__(image_bgr, conf, imgsz)` trả `[result]`:
- **Preprocess:** letterbox về `imgsz×imgsz` (giữ tỉ lệ, pad 114), BGR→RGB, `/255`, NCHW float32.
  Lưu `ratio`, `pad` để map toạ độ ngược về ảnh gốc.
- **Detection (items):** output `[1, 4+nc, M]` → transpose → boxes(xywh)+scores → conf=max lớp →
  lọc `>=conf` → **NMS** (per-class hoặc agnostic, iou~0.45–0.7 cho khớp ultralytics) → đổi xywh→xyxy
  → **un-letterbox** về gốc. Trả `.boxes.cls/.conf/.xyxy`.
- **Segmentation (board_seg):** output det `[1, 4+nc+32, M]` + proto `[1,32,mh,mw]` →
  decode det như trên, lấy 32 coeff mỗi box → `mask = sigmoid(coeff @ proto)` (mh×mw) → upsample
  imgsz → crop theo box → threshold 0.5 → `cv2.findContours` lấy polygon lớn nhất → un-letterbox.
  Trả `.masks.xy` (list polygon gốc) + `.boxes.cls`.
- Tham chiếu chuẩn: ultralytics `ops.non_max_suppression`, `ops.process_mask`, `ops.scale_boxes`,
  `ops.scale_coords` (copy logic numpy hoá).

### 3.3 Gắn shim vào code
`item_detector.load_model` + `board_segmenter.load_model`: nếu path `.onnx` → `OnnxYOLO(...)`
thay `YOLO(...)`. (Sửa tối thiểu; bỏ `.cpu().numpy()` đọc numpy thẳng.)

### 3.4 Lean image (Dockerfile.cpu)
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi==0.115.6 "uvicorn[standard]==0.32.1" python-multipart==0.0.20 \
    onnxruntime==1.19.2 opencv-python-headless==4.10.0.84 numpy==2.1.3
COPY . /app/boarddetection/
EXPOSE 8001
CMD ["uvicorn","boarddetection.server:app","--host","0.0.0.0","--port","8001"]
```
(KHÔNG torch/ultralytics → ~400MB. ONNX 1 luồng có thể set `sess_options.intra_op_num_threads`.)

---

## 4. ⭐ Verify parity (BẮT BUỘC — kẻo sai FEN) — ✅ ĐÃ PASS (xem bảng ở mục TRẠNG THÁI trên)
Bộ test có sẵn: **`C:\Resources\xqrecognition\test\bench`** + `test/ground_truth.txt` (+ các
`eval_*.log`, `detect.py` baseline torch). Quy trình:
1. Chạy pipeline bản **torch** trên bench → ghi số "exact-FEN" baseline (đã có trong `eval_*.log`).
2. Chạy pipeline bản **ONNX** trên cùng bench → so từng FEN với ground_truth.
3. **Đạt = số exact-FEN ONNX ≈ torch** (không tụt). Lệch → soi NMS/iou, mask threshold, letterbox pad,
   conf thresholds (`settings.py`: PIECE 0.25, BOARD 0.5/seg 0.25). Seg lệch vài px → đổ FEN sai → test kỹ.

---

## 5. Deploy lên VPS (sau khi parity OK) — ⏳ CÒN LẠI (user tự làm: SSH VPS + Cloudflare)
1. `docker-compose.cpu.yml`: bỏ `gpus: all`, bỏ service `cloudflared`, image build từ `Dockerfile.cpu`,
   `ports: "127.0.0.1:8001:8001"`, mount `models/` (gồm .onnx), `OCR_SHARED_SECRET` khớp portal.
2. Ship `boarddetection/` + `models/*.onnx` lên VPS → `docker compose -f docker-compose.cpu.yml up -d` →
   `curl 127.0.0.1:8001/health` + `/detect` (kèm `X-OCR-Secret`) bằng ảnh bench.
3. **Portal Flow A:** đổi `.env.production` `OCR_BACKEND_URL=http://127.0.0.1:8001` → `pm2 restart ctns-portal`.
4. **App Flow B (giữ app nguyên):** thêm nginx VPS `server_name ocr-gpu.abcxq.app;` → `proxy_pass
   http://127.0.0.1:8001;` (client_max_body_size 10M). `OCR_PUBLIC_URL` giữ `https://ocr-gpu.abcxq.app/detect`.
5. **Cloudflare (thủ công):** gỡ public hostname `ocr-gpu.abcxq.app` khỏi tunnel `ocr-gpu` (máy nhà) →
   đổi DNS `ocr-gpu` thành **A record → 103.175.146.124 (proxied)**. SSL: Cloudflare proxy (hoặc certbot).
6. Verify `https://ocr-gpu.abcxq.app/health` + detect thật từ app → tắt OCR máy nhà (`docker compose down`).

> Lưu vào memory đã có: [[ocr-gpu-migration]] — trước chỉ chờ set OCR_BACKEND_URL + secret; giờ là dời hẳn về VPS CPU.

---

## 6. Tóm tắt rủi ro / quyết định
- **Tốc độ:** CPU ~1s. Chốt chấp nhận trước khi làm.
- **Accuracy:** rủi ro nằm ở **seg mask decode** (polygon phải khớp ultralytics). Verify bench là chốt.
- **Lười rủi ro hơn:** nếu chỉ muốn lên VPS nhanh, có thể chạy **torch-CPU + ultralytics** (0 sửa code,
  image to, ~1–2s) trước; ONNX tối ưu sau. Pure-ONNX = gọn/nhanh hơn nhưng tốn công + phải test parity.
