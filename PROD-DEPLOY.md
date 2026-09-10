# PROD deploy — OCR nhận diện bàn cờ (xqdetection)

> Repo này giờ track **cả 2**: (1) **training** (models, data, weekly active-learning) và (2) **prod serving source** (`boarddetection/` + lớp deploy dưới đây). Trước kia prod source bị vendored trong `pikafishcloudengine/ocr/boarddetection` — đã gỡ, nguồn chính thức là repo này.

## Chạy ở đâu
- **portal01** (`103.175.146.124`, VPS cờ tướng) — container `xqdetection` tại `/root/apps/xqdetection`.
- CPU/ONNX (torch-free), expose qua Cloudflare tunnel **`xqdetection.abcxq.app`** (không mở port LAN).
- Không GPU nữa (máy GPU nhà đã decommission) — chạy `onnx_backend.py` + weights ONNX.

## Deploy từ đâu
- **Repo:** `github.com/cotuongbaodien/xqrecognition`
- **Nhánh serving:** `refactor/single-items-model` (nhánh active, mới nhất — chứa `boarddetection/` + deploy).
- Bản trên portal01 hiện là **copy** của cây này (verify identical: 14/14 file `boarddetection/*.py` khớp byte-for-byte, 2026-07-11).

## Thành phần deploy (ở repo root)
| File | Vai trò |
|---|---|
| `docker-compose.cpu.yml` | **compose prod** (CPU/ONNX) — cái portal01 đang dùng |
| `docker-compose.yml` | compose GPU (local/dev) |
| `boarddetection/server.py` | FastAPI `/detect` — auth `_verify_token` (JWT payload) + `X-OCR-Secret` relay + **ingest Flow B** (gom ảnh+kết quả về portal R2/ocr_logs để train) |
| `ingest_cron.py` | cron đêm đẩy `ingest_queue/` → portal `/api/ocr/ingest` rồi xoá |
| `.env.example` | mẫu env — copy thành `.env` (chmod 600) trên server, **KHÔNG commit `.env`** |
| `docs/xqdetection-vps-cpu.md`, `docs/onnx-cpu-vps-handoff.md` | doc migrate GPU→CPU/ONNX trên VPS |

## Quy trình deploy (tóm tắt)
1. Sửa code ở nhánh này, commit + push.
2. Ship cây (hoặc `boarddetection/` + compose) lên `/root/apps/xqdetection` trên portal01.
3. `docker compose -f docker-compose.cpu.yml up -d --build` (hoặc rebuild image `xqdetection:latest`) → restart container.
4. Verify: `curl` `/health` nội bộ + `xqdetection.abcxq.app` qua tunnel.

> Chi tiết migrate CPU/ONNX + luồng ingest: xem `ocr-flow-debug.md` và `docs/xqdetection-vps-cpu.md`.

## Env điều chỉnh chất lượng đọc (đổi được KHÔNG cần build lại image)

| Env | Mặc định trong code | Ý nghĩa |
|---|---|---|
| `OCR_MIN_CONFIDENCE` | **0.25** (từ 2026-09-10; trước là 0.35) | Ngưỡng conf quân cờ, khớp `settings.PIECE_CONFIDENCE_THRESHOLD` đã sweep trên bench |
| `OCR_TWO_PASS` | **1** (bật) | Đọc lượt hai bằng ảnh xoay 180° khi lượt một thiếu tướng / quá ít quân |
| `OCR_MIN_PIECES` | 5 | Số quân tối thiểu để coi là đọc được |

Cả ba đều coi chuỗi rỗng là "chưa đặt" (`server.py::_env`), nên
`- FOO=${FOO:-}` trong compose không làm chết container.

**Rollback nhanh** (không build lại, chỉ `docker compose … up -d` cho nạp env):
```
OCR_TWO_PASS=0
OCR_MIN_CONFIDENCE=0.35
```
→ hành vi hệt bản trước 2026-09-10. Rollback code hẳn: tag `prod-stable-2026-09-10`
(`git checkout prod-stable-2026-09-10 -- boarddetection/`), bản copy để ship tay nằm ở
`backups/prod_2026-09-10_e207fff/` (không track git).

### Số đo trước khi bật (2026-09-10, 600 ảnh prod thật + bench 243 có GT)

| | trước (1 lượt, conf .35) | sau (2 lượt, conf .25) |
|---|---|---|
| 600 ảnh prod qua cổng | 570 (95,0%) | **582 (97,0%)** |
| bench, ảnh chụp thẳng | 229/243 | **232/243** |
| bench, bàn lật (đen ở dưới) | 221/243 | **223/243** |
| lượt hai thực sự chạy | — | 1,7% số request |

Lượt hai CHỈ chạy khi lượt một trượt cổng ⇒ ~98% request giữ nguyên độ trễ.
Khi kết quả đến từ lượt hai, log ghi `rot180=True` và `dataset_saver` lưu **ảnh đã
xoay** (nếu không, nhãn YOLO lệch 180° so với ảnh và đầu độc vòng retrain).

> ⚠️ Kiểm tra khi ship: bản trên portal01 có `_snap_general` trong
> `boarddetection/rules_validator.py` chưa (fix 380405b, 17/08 — tướng nằm ngoài khung
> thì snap chứ không xoá). Bản mirror local `../ocr-gpu-service` **chưa có**; đo lại
> ngày 2026-09-10: fix cứu 3/600 ảnh, hỏng 0.
