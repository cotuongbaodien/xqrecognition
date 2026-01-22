# Xiangqi Recognition System

Hệ thống nhận diện bàn cờ tướng (Xiangqi/Chinese Chess) từ ảnh sử dụng Computer Vision và Deep Learning, xuất ra FEN notation.

A computer vision system for recognizing Xiangqi (Chinese Chess) board positions from images and generating FEN (Forsyth-Edwards Notation) strings.

## Features

- **Board Detection**: Nhận diện grid 9x10 của bàn cờ sử dụng YOLOv8-Segmentation
- **Piece Detection**: Nhận diện và phân loại 14 loại quân cờ sử dụng YOLOv8
- **FEN Generation**: Chuyển đổi trạng thái bàn cờ thành FEN notation chuẩn
- **REST API**: FastAPI web service để tích hợp dễ dàng
- **CLI Tools**: Command-line interfaces cho training và detection

## Documentation / Tài Liệu

| Document | Mô tả |
|----------|-------|
| [USAGE.md](docs/USAGE.md) | Hướng dẫn sử dụng chi tiết (Tiếng Việt) |
| [ACCURACY_ANALYSIS.md](docs/ACCURACY_ANALYSIS.md) | Phân tích độ chính xác & giải pháp cải thiện |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kiến trúc hệ thống chi tiết |

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd xqrecognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Datasets

```bash
# Extract datasets from zip files
python train.py setup
```

### 3. Train Models

```bash
# Train pieces detection model
python train.py pieces --epochs 100

# Train board segmentation model (optional)
python train.py board --epochs 100
```

### 4. Run Detection

```bash
# Detect from image
python detect.py --image board.jpg --output output/

# Start API server
python app.py
```

## System Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│   Board Detection           │  YOLOv8-Seg → 90 intersection points
│   (Optional)                │  → Build 9x10 grid
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Piece Detection           │  YOLOv8 → Detect 14 piece classes
│   (Required)                │  → Bounding boxes + classes
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Grid Mapping              │  Map pieces to grid positions
│                             │  → 10x9 board matrix
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   FEN Generation            │  Convert to FEN string
└─────────────┬───────────────┘
              │
              ▼
Output: "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
```

## Usage

### CLI Detection

```bash
# Single image
python detect.py --image board.jpg

# Directory of images
python detect.py --dir images/ --output results/

# With custom confidence threshold
python detect.py --image board.jpg --confidence 0.3

# Without board detection (faster, uses interpolation)
python detect.py --image board.jpg --no-board
```

### CLI Training

```bash
# Train pieces model
python train.py pieces --epochs 100 --batch-size 16

# Train board model
python train.py board --epochs 100 --batch-size 8

# Train both
python train.py all --epochs 100

# Resume training
python train.py pieces --resume
```

### API Server

```bash
# Start server
python app.py --host 0.0.0.0 --port 8000
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/detect` | POST | Detect and return JSON |
| `/detect/visualize` | POST | Return visualization image |
| `/docs` | GET | Swagger documentation |

**Example API Call:**

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@board.jpg"
```

```json
{
  "fen": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR",
  "pieces": [...],
  "piece_count": 32,
  "confidence": 0.95
}
```

### Python API

```python
from src.pipeline import XiangqiRecognizer

# Initialize
recognizer = XiangqiRecognizer(
    pieces_model_path="models/pieces_det.pt",
    use_board_detection=True
)

# Recognize
result = recognizer.recognize("board.jpg", visualize=True)

print(f"FEN: {result.fen}")
print(f"Pieces: {len(result.pieces)}")
print(f"Confidence: {result.confidence:.2%}")
```

## Project Structure

```
xqrecognition/
├── config/
│   └── settings.py          # Configuration constants
├── src/
│   ├── board_detector.py    # Board grid detection
│   ├── piece_detector.py    # Chess piece detection
│   ├── fen_generator.py     # FEN generation
│   └── pipeline.py          # Main pipeline
├── scripts/
│   ├── setup_data.py        # Dataset extraction
│   ├── train_board.py       # Board model training
│   ├── train_pieces.py      # Pieces model training
│   └── evaluate.py          # Evaluation
├── docs/
│   ├── USAGE.md             # Usage guide
│   ├── ACCURACY_ANALYSIS.md # Accuracy analysis
│   └── ARCHITECTURE.md      # System architecture
├── models/                   # Trained models
├── data/                     # Datasets
├── app.py                   # FastAPI server
├── train.py                 # Training CLI
├── detect.py                # Detection CLI
├── requirements.txt
├── Dockerfile
└── README.md
```

## Chess Pieces (14 Classes)

| ID | Name | Tiếng Việt | FEN |
|----|------|------------|-----|
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

## FEN Notation

**Standard starting position:**
```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
```

- Row 0 (top): Black's back rank
- Row 9 (bottom): Red's back rank
- `/` separates rows
- Numbers = consecutive empty squares
- Uppercase = Red, lowercase = Black

## Accuracy & Improvement

Xem chi tiết tại [ACCURACY_ANALYSIS.md](docs/ACCURACY_ANALYSIS.md)

### Expected Accuracy

| Stage | Target Accuracy |
|-------|-----------------|
| Piece Detection (mAP@50) | 90-95% |
| FEN Exact Match | 75-85% |
| Piece Position Accuracy | 95-98% |

### Key Improvement Strategies

1. **Data Augmentation**: Rotation, perspective, lighting variations
2. **Game Rules Validation**: Filter invalid positions
3. **Ensemble Models**: Combine multiple YOLO models
4. **Hybrid Board Detection**: ML + Traditional CV

## Docker

```bash
# Build
docker build -t xqrecognition .

# Run
docker run -p 8000:8000 xqrecognition
```

## Evaluation

```bash
# Evaluate pieces model
python scripts/evaluate.py pieces --split test

# Evaluate full pipeline
python scripts/evaluate.py pipeline \
  --test-dir test_images/ \
  --ground-truth ground_truth.json \
  --output results.json
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLOv8
- OpenCV
- FastAPI
- CUDA (optional, for GPU acceleration)

## License

MIT License

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request
