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
