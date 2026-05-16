# Hướng dẫn Backend Integration — Xiangqi Recognition API

Tài liệu này mô tả cách backend team integrate API nhận diện bàn cờ tướng → FEN.

## Tổng quan

Service nhận **1 ảnh bàn cờ** (JPG/PNG/WebP) và trả về **FEN notation** + danh sách quân cờ detect được.

- Pipeline: 1 model YOLO (`items.pt`) detect 18 classes (14 quân + 4 landmarks) → dựng grid 9x10 → map quân vào ô → output FEN.
- 1 forward pass, không cần model phụ.

## API Endpoints

Server FastAPI mặc định chạy trên port `8000`. Tất cả endpoints bên dưới relative đến base URL (vd `https://xq.example.com`).

### `POST /detect`

Endpoint chính — upload ảnh, nhận JSON kết quả.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: 1 file field tên `file` (ảnh JPG/PNG/WebP)

**Example (curl):**
```bash
curl -X POST "https://xq.example.com/detect" \
  -F "file=@/path/to/board.jpg"
```

**Example (JavaScript fetch):**
```javascript
const formData = new FormData();
formData.append('file', imageFile);  // imageFile từ <input type="file"> hoặc Blob

const res = await fetch('https://xq.example.com/detect', {
  method: 'POST',
  body: formData,
});
const data = await res.json();
console.log(data.fen);
```

**Example (Python requests):**
```python
import requests
with open('board.jpg', 'rb') as f:
    res = requests.post('https://xq.example.com/detect', files={'file': f})
data = res.json()
print(data['fen'])
```

**Response (200 OK):**
```json
{
  "fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
  "pieces": [
    {
      "class_id": 13,
      "class_name": "red-chariot",
      "display_name": "Xe đỏ",
      "confidence": 0.95,
      "bbox": [120.5, 240.1, 180.3, 300.4],
      "center": [150.4, 270.2],
      "fen_symbol": "R"
    }
    // ... mỗi quân detect được là 1 object như trên
  ],
  "piece_count": 32,
  "confidence": 0.94,
  "errors": []
}
```

| Field | Type | Mô tả |
|---|---|---|
| `fen` | string | FEN notation 10 hàng cách nhau bằng `/`. Hàng 0 = đen (top), hàng 9 = đỏ (bottom). |
| `pieces` | array | Chi tiết từng quân detect được. |
| `pieces[].class_name` | string | Tên class kebab-case (vd `red-chariot`, `black-cannon`). |
| `pieces[].display_name` | string | Tên tiếng Việt (vd `Xe đỏ`, `Pháo đen`). |
| `pieces[].fen_symbol` | string | Ký tự FEN (`R/N/B/A/K/C/P` đỏ, `r/n/b/a/k/c/p` đen). |
| `pieces[].bbox` | [x1,y1,x2,y2] | Bounding box trong ảnh gốc (pixel). |
| `pieces[].center` | [cx,cy] | Tâm quân cờ. |
| `pieces[].confidence` | float | Độ tin cậy detection (0-1). |
| `piece_count` | int | Tổng số quân detect được (max 32). |
| `confidence` | float | Trung bình confidence của tất cả quân. |
| `errors` | array | Cảnh báo nếu có (vd grid build thất bại, dùng fallback). |

**Response (4xx/5xx):**
```json
{
  "detail": "Could not load image: file is not a valid image"
}
```

### `POST /detect/visualize`

Trả về ảnh PNG có grid + bbox overlay (để debug, kiểm tra trực quan).

**Request:** giống `/detect`
**Response:** `image/png` binary

```bash
curl -X POST "https://xq.example.com/detect/visualize" \
  -F "file=@board.jpg" \
  --output result.png
```

### `GET /health`

Health check để monitoring/load balancer kiểm tra service alive.

**Response:**
```json
{
  "status": "healthy",
  "items_model_loaded": true
}
```

### `GET /docs`

FastAPI tự generate Swagger UI. Truy cập trên browser để test API trực tiếp:
```
https://xq.example.com/docs
```

## Format FEN

FEN cờ tướng (Xiangqi):
- **10 hàng** cách nhau `/`, từ hàng 0 (đen, top) đến hàng 9 (đỏ, bottom).
- **Mỗi hàng** mô tả 9 cột (a-i từ trái sang phải).
- **Số** = số ô trống liên tiếp (vd `5` = 5 ô trống).
- **Chữ hoa** = quân đỏ, **chữ thường** = quân đen.

| Quân | Đỏ | Đen | Tiếng Việt |
|---|---|---|---|
| Xe (Chariot/Rook) | `R` | `r` | Xe |
| Mã (Horse) | `N` | `n` | Mã |
| Tượng (Elephant) | `B` | `b` | Tượng |
| Sĩ (Advisor) | `A` | `a` | Sĩ |
| Tướng (General) | `K` | `k` | Tướng |
| Pháo (Cannon) | `C` | `c` | Pháo |
| Tốt (Soldier/Pawn) | `P` | `p` | Tốt |

**Vị trí khởi đầu chuẩn:**
```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
```

## Lưu ý

### Yêu cầu ảnh đầu vào

| Tiêu chí | Khuyến nghị |
|---|---|
| Format | JPG, PNG, WebP, BMP |
| Resolution | ≥ 640x640 |
| Bàn cờ visibility | Toàn bộ bàn cờ trong khung |
| Bàn cờ chiếm | ≥ 50% diện tích ảnh |
| Góc chụp | Thẳng đứng hoặc nghiêng nhẹ < 15° |
| Orientation | **Portrait** (board cao hơn rộng) — board landscape có thể FEN sai |
| Max file size | Khuyến nghị < 10MB |

### Limitations hiện tại (cần biết)

1. **Mirror trái-phải**: bàn cờ Xiangqi đối xứng → đôi lúc FEN bị mirror trái-phải. Nếu app consuming có chức năng mirror, có thể tự xử lý. Server hiện KHÔNG auto mirror.
2. **Board rotation 90°**: ảnh chụp board nằm ngang (landscape) → FEN sẽ bị rotate. Khuyến nghị bắt buộc user chụp portrait.
3. **Cold start ~5-10s**: load model lần đầu. Service đã pre-warm nên request đầu tiên cũng nhanh.
4. **No auth/rate-limit**: chưa có. Nếu API public, backend team cần thêm middleware (API key, JWT, IP whitelist...).

### Performance

- GPU (RTX 3060): ~50-100ms/request
- CPU only: ~500ms-2s/request

### Error handling phía consumer

Backend nên handle các trường hợp:
- **400**: file không phải ảnh, file quá lớn
- **500**: model crash, image too small
- **`piece_count < 5` hoặc `confidence < 0.5`**: FEN có thể không reliable → cảnh báo user chụp lại
- **`errors` array có giá trị**: grid build có vấn đề → vẫn có FEN nhưng kém chính xác

## Deploy server

### Option 1: Python trực tiếp

```bash
git clone https://github.com/cotuongbaodien/xqrecognition.git
cd xqrecognition
git checkout main  # hoặc branch refactor/single-items-model

# Cài dependencies
pip install -r requirements.txt

# QUAN TRỌNG: copy file model items.pt (~22MB) vào models/
# File này KHÔNG có trong git (gitignored).
# Lấy từ dev environment hoặc cloud storage.

# Production run với uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

### Option 2: Docker

```bash
# Build
docker build -t xqrecognition .

# Run với GPU
docker run -d -p 8000:8000 --gpus all \
  -v /path/to/models:/app/models \
  --name xq xqrecognition

# Run CPU-only
docker run -d -p 8000:8000 \
  -v /path/to/models:/app/models \
  --name xq xqrecognition
```

### Option 3: Behind nginx + SSL

```nginx
server {
    listen 443 ssl http2;
    server_name xq.example.com;

    # SSL config...

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;  # giới hạn upload
        proxy_read_timeout 30s;
    }
}
```

### Yêu cầu hệ thống

| Resource | Minimum | Recommended |
|---|---|---|
| Python | 3.8+ | 3.10+ |
| RAM | 4GB | 8GB |
| Disk | 2GB | 5GB |
| GPU | Optional | NVIDIA với CUDA 11.8+ |
| OS | Linux/Windows/Mac | Linux (Ubuntu 22.04+) |

## Liên hệ / Vấn đề

- Repo: https://github.com/cotuongbaodien/xqrecognition
- Branch hiện tại đang work tốt: `refactor/single-items-model`
- File model: `models/items.pt` (lấy từ dev/cloud storage, gitignored)
