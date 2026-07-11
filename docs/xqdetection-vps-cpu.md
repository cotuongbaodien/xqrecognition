# xqdetection — OCR CPU/ONNX trên VPS (vận hành + kiến trúc)

> OCR nhận diện bàn cờ (YOLO11) đã **dời từ máy-nhà-GPU sang VPS portal** chạy **CPU bằng
> ONNX Runtime**, expose qua tunnel **`xqdetection.abcxq.app`**. Tài liệu này = trạng thái
> đang chạy prod + runbook. Migration/handoff gốc: [`onnx-cpu-vps-handoff.md`](./onnx-cpu-vps-handoff.md).

## Kiến trúc đang chạy
```
App ──Flow B (token)──► xqdetection.abcxq.app ──Cloudflare tunnel──► xqdetection:8001 (VPS, ONNX CPU)
                                                                       │ trả FEN ngay (~1.4s e2e)
                                                                       └─(background) ghi QUEUE ra đĩa
VPS host: cron 3h sáng ──► đọc queue ──► POST portal /api/ocr/ingest ──► R2 + ocr_logs (review) ──► XOÁ queue
Portal (PM2, cùng VPS) ──Flow A relay──► xqdetection.abcxq.app/detect   (app dùng Flow B là chính)
```
- **VPS**: `103.175.146.124` · 8 core · 16GB · KHÔNG GPU. OCR + portal + engine cùng máy.
- **Tách hẳn portal**: container Docker riêng, repo riêng (`ocr-gpu-service` / canonical
  `xqrecognition/boarddetection`). Tương lai overload → dời server khác chỉ đổi env/DNS.

## Thành phần (`/root/apps/xqdetection/` trên VPS)
| | Mô tả |
|---|---|
| container `xqdetection` | uvicorn FastAPI :8001, 1 worker, model ONNX (`items.onnx` 960 + `board_seg.onnx` 640) |
| container `xqdetection-cloudflared` | connector tunnel `xqdetection.abcxq.app` → `xqdetection:8001` |
| `boarddetection/ingest_queue/` | queue ảnh Flow B chờ cron (bind-mount host) |
| `ingest_cron.py` + crontab `0 3 * * *` | đẩy queue → portal → R2/ocr_logs → clear |
| `.env` (KHÔNG commit) | `OCR_SHARED_SECRET`, `CLOUDFLARE_TUNNEL_TOKEN`, `OCR_MODEL_FORMAT=onnx`, `OCR_MODEL_VERSION` |

## 2 luồng
- **Flow B (app, chính):** app xin token ở portal `/api/ocr/token` (lấy trước, không tính giờ) →
  upload ảnh thẳng `xqdetection.abcxq.app/detect` (header `X-OCR-Token`) → FEN. OCR ghi ảnh ra
  **queue** (archive ngầm). Portal **không** nằm trên đường ảnh.
- **Flow A (relay):** portal `/api/ocr/board-to-fen` → relay `OCR_BACKEND_URL/detect` (header
  `X-OCR-Secret`). Portal tự lưu R2/ocr_logs. (App hiện không dùng.)
- Portal `.env.production`: `OCR_BACKEND_URL=OCR_PUBLIC_URL=https://xqdetection.abcxq.app(/detect)`.

## Concurrency + tốc độ (đã tối ưu)
- **Inference chạy threadpool** (`run_in_threadpool`) → onnxruntime nhả GIL → nhiều request
  detect **song song**, không chặn event loop. (Trước: blocking → request thứ 2 chờ → timeout.)
- **NMS thread-safe**: `_class_aware_nms(xyxy, cls, score, iou)` nhận `cls` qua THAM SỐ (bỏ
  `self._cls_off` shared mutable) → reentrant khi 2 request đồng thời.
- **Graph-opt**: `ORT_ENABLE_ALL`.
- Số đo: OCR `infer_ms` ~0.4s (1 request) / ~0.7–0.8s (2 song song, chia CPU); end-to-end ~1.4s
  (chủ yếu **upload ảnh** — app gửi ~300KB). GPU cũ <50ms; CPU chậm hơn nhưng OCR ít dùng → ok.
- Log mỗi detect: `auth=… detected=… conf=… pieces=… size_kb=… infer_ms=… total_ms=…`.

## Archive: queue-to-disk + cron (KHÔNG real-time)
- `/detect` (Flow B) ghi `<id>.jpg` + `<id>.json` (meta uid/dev/fen/conf/detected/ms) vào
  `ingest_queue/` — **luôn thành công, tách khỏi request** (lỗi archive KHÔNG ảnh hưởng FEN/app).
- `ingest_cron.py` (3h sáng): mỗi item → POST portal `/api/ocr/ingest` (localhost:3100,
  `X-OCR-Secret`) → portal lưu **R2 + ocr_logs** (admin gallery review) → **xoá file** đã up
  (đĩa nhẹ). Lỗi → giữ lại retry lần sau. Dedup theo hash ở portal.
- Endpoint portal: `src/app/api/ocr/ingest/route.ts` (repo cotuongnghiasing_portal).

**Chạy ingest THỦ CÔNG** (đẩy queue lên ngay, không đợi 3am — vd muốn review liền):
```bash
ssh root@103.175.146.124 'set -a; . /root/apps/xqdetection/.env; set +a; \
  python3 /root/apps/xqdetection/ingest_cron.py'
# In ra: uploaded=N failed=M. Retry-safe + idempotent → CHẠY LẠI để dọn nốt phần
# failed (portal đôi khi refuse 1–2 cái khi spike memory). failed = file còn ở queue.
ssh root@103.175.146.124 'ls /root/apps/xqdetection/boarddetection/ingest_queue/'  # xem queue chờ
```

## RUNBOOK
**Đổi code OCR** (code bind-mount → khỏi rebuild image):
```bash
# local: sửa boarddetection/*.py
scp ... boarddetection/server.py  root@VPS:/root/apps/xqdetection/boarddetection/
ssh VPS "docker restart xqdetection && docker restart xqdetection-cloudflared"   # ⚠️ phải restart cả 2
```
> ⚠️ **Restart/recreate container OCR LÀM RỚT cloudflared origin → public 000/timeout.**
> Luôn `docker restart xqdetection-cloudflared` ngay sau khi restart OCR.

**Đổi model** (retrain): export `.pt`→`.onnx` (xem handoff §3.1) → thay `models/*.onnx` →
`docker restart xqdetection && docker restart xqdetection-cloudflared`. (Nhớ chạy parity bench.)

**Build image (khi đổi deps)**: build **LOCAL** (`docker build -f boarddetection/Dockerfile.cpu
-t xqdetection:latest ./boarddetection`) → `docker save | gzip | ssh "docker load"` →
`docker compose -f docker-compose.cpu.yml up -d --no-build` → **restart cloudflared**. KHÔNG build trên VPS.

**Logs / kiểm tra:**
```bash
docker logs xqdetection --tail 20            # detect + size_kb/infer_ms/total_ms
docker logs xqdetection-cloudflared | grep -i registered
cat /root/apps/xqdetection/ingest_cron.log   # kết quả cron
ls /root/apps/xqdetection/boarddetection/ingest_queue/   # queue đang chờ
```

**Verify nhanh:**
```bash
curl https://xqdetection.abcxq.app/health
# detect (Flow A secret): curl -X POST .../detect -H "X-OCR-Secret: <secret>" -F image=@board.jpg
```

**Revert về máy nhà** (nếu sự cố): sửa portal `.env.production` `OCR_BACKEND_URL`/`OCR_PUBLIC_URL`
→ `https://ocr-gpu.abcxq.app` → `pm2 restart ctns-portal`. (Máy nhà GPU vẫn chạy tới khi decommission.)

## Quyết định / lưu ý
- **"1 fail cùng ảnh"** trước đây = **token 1 giây/device** (`/api/ocr/token` rate-limit), KHÔNG
  phải OCR đơ. Quét 2 lần sát nhau cùng máy → lần 2 xin token bị 429.
- **Ingest fail KHÔNG làm hỏng scan** (background, sau khi đã trả FEN).
- **Decommission máy nhà**: khi ổn định → stop `cloudflared` + OCR ở `ocr-gpu-service` máy nhà.

## ⭐ Cần PORT về canonical `xqrecognition/boarddetection` (kẻo sync đè mất)
| File | Thay đổi |
|---|---|
| `boarddetection/server.py` | threadpool inference; queue-to-disk (`_queue_save`, `OCR_QUEUE_DIR`); log `size_kb/total_ms`; (đã có auth token + `_ingest` cũ) |
| `boarddetection/onnx_backend.py` | graph-opt `ORT_ENABLE_ALL`; NMS thread-safe (`_class_aware_nms` nhận `cls`) |
| `boarddetection/Dockerfile.cpu` | image lean onnxruntime (đã port trước) |
| `docker-compose.cpu.yml` | stack CPU + cloudflared + bind-mount code + extra_hosts |
| `ingest_cron.py` | cron đẩy queue → portal |
| `src/app/api/ocr/ingest/route.ts` (repo portal) | endpoint nhận ingest |
