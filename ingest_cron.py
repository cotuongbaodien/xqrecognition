#!/usr/bin/env python3
"""Cron đêm: đẩy queue ảnh OCR (Flow B) lên portal /api/ocr/ingest (R2 + ocr_logs
để review), rồi XOÁ file đã up cho nhẹ đĩa. Chạy trên VPS host (KHÔNG trong container).

  Queue do server.py ghi ra: <QUEUE>/<id>.jpg + <id>.json (meta uid/dev/fen/...).
  Mỗi item POST multipart về portal localhost:3100. HTTP 200 → xoá cả .jpg + .json.
  Lỗi/không-200 → giữ lại cho lần chạy sau (retry). Idempotent (portal dedup theo hash).

Crontab (0h... 3h sáng):
  0 3 * * * set -a; . /root/apps/xqdetection/.env; set +a; /usr/bin/python3 \
    /root/apps/xqdetection/ingest_cron.py >> /root/apps/xqdetection/ingest_cron.log 2>&1
"""
import glob
import io
import json
import os
import sys
import time
import urllib.request
import uuid
from datetime import datetime

# Đường dẫn HOST (không phải trong container).
QUEUE = "/root/apps/xqdetection/boarddetection/ingest_queue"
# Portal nội bộ (host → next-server :3100). Cron chạy trên host nên dùng 127.0.0.1.
PORTAL = "http://127.0.0.1:3100/api/ocr/ingest"
SECRET = os.environ.get("OCR_SHARED_SECRET", "")
TIMEOUT = 30
MAX_PER_RUN = 5000  # chặn lỡ queue phình bất thường
# Đẩy theo BATCH nhỏ + nghỉ giữa batch → portal xử từ tốn, KHÔNG vọt memory chạm
# max_memory_restart (1GB) rồi restart (gây gián đoạn live + ingest refused). Mỗi
# ingest nặng (sharp + 2 R2 upload). Drain hết queue nhưng theo nhịp.
BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "10"))
BATCH_PAUSE_SEC = float(os.environ.get("INGEST_BATCH_PAUSE_SEC", "4"))
ITEM_SLEEP_SEC = float(os.environ.get("INGEST_ITEM_SLEEP_SEC", "0.3"))


def post(jpg_path: str, meta: dict) -> int:
    boundary = "----xqcron" + uuid.uuid4().hex
    buf = io.BytesIO()
    for k, v in meta.items():
        buf.write(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        )
    buf.write(
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
        f'filename="board.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
    )
    with open(jpg_path, "rb") as f:
        buf.write(f.read())
    buf.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        PORTAL,
        data=buf.getvalue(),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-OCR-Secret": SECRET,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status


def main() -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    if not SECRET:
        print(f"{ts} ingest_cron: THIẾU OCR_SHARED_SECRET, bỏ qua")
        sys.exit(1)
    if not os.path.isdir(QUEUE):
        print(f"{ts} ingest_cron: queue trống ({QUEUE})")
        return
    ok = fail = 0
    files = sorted(glob.glob(os.path.join(QUEUE, "*.json")))[:MAX_PER_RUN]
    for i, js in enumerate(files):
        # Nghỉ giữa mỗi batch để portal GC bộ nhớ (sharp/R2) → không vọt memory.
        if i > 0 and i % BATCH_SIZE == 0:
            time.sleep(BATCH_PAUSE_SEC)
        jpg = js[:-5] + ".jpg"
        try:
            if not os.path.exists(jpg):
                os.remove(js)  # meta mồ côi → bỏ
                continue
            with open(js, encoding="utf-8") as f:
                meta = json.load(f)
            status = post(jpg, meta)
            if status == 200:
                os.remove(jpg)
                os.remove(js)
                ok += 1
            else:
                fail += 1
            time.sleep(ITEM_SLEEP_SEC)  # throttle nhẹ trong batch
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"{ts} ingest_cron: lỗi {os.path.basename(js)}: {e}")
    print(f"{ts} ingest_cron: uploaded={ok} failed={fail} (giữ lại retry)")


if __name__ == "__main__":
    main()
