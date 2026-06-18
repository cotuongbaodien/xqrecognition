# Deploy `boarddetection` as Python OCR service

## Prerequisites trên prod

- Python 3.11+ (`dnf install python3.11 python3.11-pip python3.11-devel`)
- Đủ RAM: ~1.5GB peak khi inference. Mở swap thêm nếu RAM siết.
- Disk: ~2GB cho venv (torch CPU ~700MB + ultralytics + opencv)
- Model file `boarddetection/models/items.pt` (~22MB) — copy thủ công, **KHÔNG commit git**

## Setup lần đầu

```bash
cd /root/apps/cotuongnghiasing_portal/boarddetection
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python -c "from boarddetection import XiangqiRecognizer; r=XiangqiRecognizer(); print('OK')"
```

## Run dev

```bash
source .venv/bin/activate
cd /root/apps/cotuongnghiasing_portal     # parent của boarddetection
uvicorn boarddetection.server:app --host 127.0.0.1 --port 8001 --log-level info
```

## Run production qua PM2

```bash
pm2 start /root/apps/cotuongnghiasing_portal/boarddetection/.venv/bin/uvicorn \
  --name ctns-ocr \
  --cwd /root/apps/cotuongnghiasing_portal \
  --max-memory-restart 1500M \
  --interpreter none \
  -- boarddetection.server:app --host 127.0.0.1 --port 8001 --log-level info
pm2 save
```

Smoke test:
```bash
curl http://127.0.0.1:8001/health
# → {"status":"ok","model":"loaded"}

curl -X POST http://127.0.0.1:8001/detect -F "image=@/path/to/board.jpg"
# → {"detected":true,"fen":"...","confidence":0.87,"pieces_count":32,...}
```

## Update model

```bash
# Stop service to release file lock
pm2 stop ctns-ocr

# Copy new items.pt (scp from dev)
scp items.pt root@103.200.20.107:/root/apps/cotuongnghiasing_portal/boarddetection/models/items.pt

# Restart
pm2 restart ctns-ocr
```

## Update code

```bash
# Pull latest portal code
cd /root/apps/cotuongnghiasing_portal
tar -xzf /tmp/ctns_portal_new.tar.gz --no-overwrite-dir
# Model + venv preserved (gitignored)

pm2 restart ctns-ocr
```

## Logs

```bash
pm2 logs ctns-ocr --lines 100
```
