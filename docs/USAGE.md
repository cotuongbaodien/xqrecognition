# Hướng Dẫn Sử Dụng - Xiangqi Recognition System

## Mục Lục

1. [Cài Đặt](#1-cài-đặt)
2. [Chuẩn Bị Dữ Liệu](#2-chuẩn-bị-dữ-liệu)
3. [Huấn Luyện Model](#3-huấn-luyện-model)
4. [Nhận Diện Bàn Cờ](#4-nhận-diện-bàn-cờ)
5. [Sử Dụng API](#5-sử-dụng-api)
6. [Đánh Giá Model](#6-đánh-giá-model)
7. [Xử Lý Lỗi Thường Gặp](#7-xử-lý-lỗi-thường-gặp)

---

## 1. Cài Đặt

### 1.1 Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB cho training)
- **GPU**: NVIDIA GPU với CUDA (tùy chọn, tăng tốc training)
- **Disk**: Tối thiểu 10GB trống

### 1.2 Cài Đặt Dependencies

```bash
# Clone repository
git clone <repository-url>
cd xqrecognition

# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 1.3 Kiểm Tra Cài Đặt

```bash
# Kiểm tra CUDA (nếu có GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Kiểm tra ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## 2. Chuẩn Bị Dữ Liệu

### 2.1 Dataset Cần Thiết

Hệ thống sử dụng 2 dataset:

| Dataset | Mục đích | File |
|---------|----------|------|
| Board Segmentation | Nhận diện điểm giao trên bàn cờ | `seg_chinese_chess.v3i.yolov8.zip` |
| Pieces Detection | Nhận diện 14 loại quân cờ | `Chinese-chess.v9i.yolov8.zip` |

### 2.2 Giải Nén Dataset

```bash
# Tự động giải nén tất cả dataset
python train.py setup
```

Hoặc thủ công:

```bash
# Giải nén board segmentation dataset
unzip download/seg_chinese_chess.v3i.yolov8.zip -d data/board_seg/

# Giải nén pieces detection dataset
unzip download/Chinese-chess.v9i.yolov8.zip -d data/pieces/
```

### 2.3 Kiểm Tra Cấu Trúc Dataset

Sau khi giải nén, cấu trúc phải như sau:

```
data/
├── board_seg/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   └── test/
└── pieces/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    └── test/
```

### 2.4 Kiểm Tra data.yaml

File `data.yaml` phải chứa đúng đường dẫn và class names:

**Pieces Detection (data/pieces/data.yaml):**
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 14
names:
  0: Advisor_black
  1: Advisor_red
  2: Cannon_black
  3: Cannon_red
  4: Elephant_black
  5: Elephant_red
  6: General_black
  7: General_red
  8: Knight_black
  9: Knight_red
  10: Pawn_black
  11: Pawn_red
  12: Rook_black
  13: Rook_red
```

---

## 3. Huấn Luyện Model

### 3.1 Huấn Luyện Pieces Detection Model

```bash
# Huấn luyện cơ bản (100 epochs)
python train.py pieces --epochs 100

# Huấn luyện với tùy chọn
python train.py pieces \
    --epochs 150 \
    --batch-size 16 \
    --img-size 640 \
    --device cuda  # hoặc cpu, mps (Mac)

# Tiếp tục huấn luyện từ checkpoint
python train.py pieces --resume
```

### 3.2 Huấn Luyện Board Segmentation Model

```bash
# Huấn luyện board segmentation
python train.py board --epochs 100 --batch-size 8

# Lưu ý: Segmentation cần nhiều VRAM hơn, giảm batch-size nếu gặp lỗi OOM
python train.py board --epochs 100 --batch-size 4
```

### 3.3 Huấn Luyện Cả Hai Model

```bash
python train.py all --epochs 100
```

### 3.4 Theo Dõi Training

Training logs được lưu tại:
- `runs/pieces_det/train/` - Pieces detection
- `runs/board_seg/train/` - Board segmentation

Xem metrics trong TensorBoard:
```bash
tensorboard --logdir runs/
```

### 3.5 Model Đầu Ra

Sau training, models được lưu tại:
- `models/pieces_det.pt` - Pieces detection model
- `models/board_seg.pt` - Board segmentation model

---

## 4. Nhận Diện Bàn Cờ

### 4.1 Nhận Diện Từ Ảnh Đơn

```bash
# Cơ bản
python detect.py --image path/to/board.jpg

# Lưu kết quả với visualization
python detect.py --image board.jpg --output output/

# Điều chỉnh confidence threshold
python detect.py --image board.jpg --confidence 0.3

# Tắt board detection (nhanh hơn, dùng interpolation)
python detect.py --image board.jpg --no-board
```

### 4.2 Nhận Diện Từ Thư Mục

```bash
# Xử lý tất cả ảnh trong thư mục
python detect.py --dir images/ --output results/
```

### 4.3 Output Format

**Console output:**
```
Detection Result
============================================================
Image: board.jpg
FEN: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
Pieces detected: 32
Confidence: 95.23%
```

**JSON output (results.json):**
```json
{
  "fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
  "pieces": [
    {
      "class_id": 12,
      "class_name": "Rook_black",
      "display_name": "Xe đen",
      "confidence": 0.95,
      "bbox": [10, 20, 50, 60],
      "center": [30, 40],
      "fen_symbol": "r"
    }
  ],
  "piece_count": 32,
  "confidence": 0.9523
}
```

### 4.4 Sử Dụng Trong Python Code

```python
from src.pipeline import XiangqiRecognizer

# Khởi tạo recognizer
recognizer = XiangqiRecognizer(
    board_model_path="models/board_seg.pt",
    pieces_model_path="models/pieces_det.pt",
    use_board_detection=True
)

# Nhận diện từ file
result = recognizer.recognize("board.jpg", visualize=True)

print(f"FEN: {result.fen}")
print(f"Pieces: {len(result.pieces)}")
print(f"Confidence: {result.confidence:.2%}")

# Lưu visualization
import cv2
if result.visualization is not None:
    cv2.imwrite("output.png", result.visualization)

# Nhận diện từ numpy array
import cv2
image = cv2.imread("board.jpg")
result = recognizer.recognize_image(image)
```

---

## 5. Sử Dụng API

### 5.1 Khởi Động Server

```bash
# Mặc định: http://localhost:8000
python app.py

# Tùy chỉnh host/port
python app.py --host 0.0.0.0 --port 8080
```

### 5.2 API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/` | GET | Thông tin API |
| `/health` | GET | Health check |
| `/detect` | POST | Nhận diện và trả về JSON |
| `/detect/visualize` | POST | Trả về ảnh visualization |
| `/detect/json-with-image` | POST | JSON + base64 image |

### 5.3 Ví Dụ Gọi API

**Với curl:**
```bash
# Nhận diện cơ bản
curl -X POST "http://localhost:8000/detect" \
  -F "file=@board.jpg"

# Lấy visualization image
curl -X POST "http://localhost:8000/detect/visualize" \
  -F "file=@board.jpg" \
  --output result.png
```

**Với Python requests:**
```python
import requests

# Upload và nhận diện
with open("board.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect",
        files={"file": f}
    )

result = response.json()
print(f"FEN: {result['fen']}")
print(f"Pieces: {result['piece_count']}")
```

**Với JavaScript/Fetch:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/detect', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log('FEN:', result.fen);
```

### 5.4 API Documentation

Truy cập Swagger UI tại: `http://localhost:8000/docs`

---

## 6. Đánh Giá Model

### 6.1 Đánh Giá Pieces Detection

```bash
python scripts/evaluate.py pieces --split test
```

Output:
```
Pieces Detection Model Evaluation
============================================================
Model: models/pieces_det.pt
Dataset: data/pieces/data.yaml
Split: test

Results:
  mAP@50: 0.9234
  mAP@50-95: 0.8567
  Precision: 0.9123
  Recall: 0.8945
```

### 6.2 Đánh Giá Board Segmentation

```bash
python scripts/evaluate.py board --split test
```

### 6.3 Đánh Giá Full Pipeline

```bash
# Cần file ground truth JSON
python scripts/evaluate.py pipeline \
    --test-dir test_images/ \
    --ground-truth ground_truth.json \
    --output evaluation_results.json
```

**Format ground_truth.json:**
```json
{
  "image1.jpg": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
  "image2.jpg": "2bakab2/9/4c4/9/9/9/9/3C5/4A4/2BAK1B2"
}
```

### 6.4 Hiểu Các Metrics

| Metric | Ý nghĩa | Giá trị tốt |
|--------|---------|-------------|
| mAP@50 | Mean Average Precision at IoU 0.5 | > 0.85 |
| mAP@50-95 | mAP trung bình từ IoU 0.5-0.95 | > 0.70 |
| Precision | Tỷ lệ dự đoán đúng / tổng dự đoán | > 0.90 |
| Recall | Tỷ lệ phát hiện được / tổng thực tế | > 0.85 |
| FEN Accuracy | Tỷ lệ FEN khớp hoàn toàn | > 0.80 |

---

## 7. Xử Lý Lỗi Thường Gặp

### 7.1 CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
```bash
# Giảm batch size
python train.py pieces --batch-size 8

# Giảm image size
python train.py pieces --img-size 416

# Sử dụng CPU
python train.py pieces --device cpu
```

### 7.2 Model Not Found

```
FileNotFoundError: Pieces model not found at models/pieces_det.pt
```

**Giải pháp:**
- Chạy training trước: `python train.py pieces`
- Hoặc copy model có sẵn vào `models/`

### 7.3 Dataset Not Found

```
FileNotFoundError: Could not find data.yaml
```

**Giải pháp:**
```bash
# Chạy setup
python train.py setup

# Hoặc kiểm tra đường dẫn trong data.yaml
```

### 7.4 Nhận Diện Không Chính Xác

**Nguyên nhân có thể:**
1. Ảnh chất lượng thấp/mờ
2. Góc chụp quá nghiêng
3. Ánh sáng không đủ
4. Model chưa được train đủ

**Giải pháp:**
1. Sử dụng ảnh chất lượng cao hơn
2. Chụp từ góc thẳng đứng
3. Đảm bảo đủ ánh sáng
4. Train thêm epochs hoặc augment data

### 7.5 FEN Không Đúng Với Thực Tế

**Kiểm tra:**
```python
from src.fen_generator import FENGenerator

fen_gen = FENGenerator()
is_valid, errors = fen_gen.validate_fen(result.fen)

if not is_valid:
    print("FEN errors:", errors)
```

---

## Phụ Lục

### A. Danh Sách Quân Cờ

| ID | Tên | Tiếng Việt | FEN |
|----|-----|------------|-----|
| 0 | Advisor_black | Sĩ đen | a |
| 1 | Advisor_red | Sĩ đỏ | A |
| 2 | Cannon_black | Pháo đen | c |
| 3 | Cannon_red | Pháo đỏ | C |
| 4 | Elephant_black | Tượng đen | b |
| 5 | Elephant_red | Tượng đỏ | B |
| 6 | General_black | Tướng đen | k |
| 7 | General_red | Tướng đỏ | K |
| 8 | Knight_black | Mã đen | n |
| 9 | Knight_red | Mã đỏ | N |
| 10 | Pawn_black | Tốt đen | p |
| 11 | Pawn_red | Tốt đỏ | P |
| 12 | Rook_black | Xe đen | r |
| 13 | Rook_red | Xe đỏ | R |

### B. FEN Notation

**Vị trí bắt đầu tiêu chuẩn:**
```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
```

**Cách đọc:**
- Hàng 0 (đỉnh): `rnbakabnr` = Xe, Mã, Tượng, Sĩ, Tướng, Sĩ, Tượng, Mã, Xe (đen)
- Số = số ô trống liên tiếp
- `/` = xuống hàng mới
- Chữ hoa = Đỏ, chữ thường = Đen
