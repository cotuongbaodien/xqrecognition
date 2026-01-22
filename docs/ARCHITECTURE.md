# Kiến Trúc Hệ Thống - Xiangqi Recognition System

## Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Cấu Trúc Dự Án](#2-cấu-trúc-dự-án)
3. [Luồng Xử Lý](#3-luồng-xử-lý)
4. [Chi Tiết Các Module](#4-chi-tiết-các-module)
5. [Data Flow](#5-data-flow)
6. [API Design](#6-api-design)
7. [Deployment](#7-deployment)

---

## 1. Tổng Quan

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Web App   │  │  Mobile App │  │   CLI Tool  │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
└─────────┼────────────────┼────────────────┼─────────────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────────┐
│                          │         API Layer                         │
│  ┌───────────────────────▼───────────────────────┐                  │
│  │              FastAPI Server                    │                  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────┐   │                  │
│  │  │ /detect │  │ /health │  │ /visualize  │   │                  │
│  │  └────┬────┘  └─────────┘  └──────┬──────┘   │                  │
│  └───────┼───────────────────────────┼──────────┘                  │
└──────────┼───────────────────────────┼──────────────────────────────┘
           │                           │
┌──────────┼───────────────────────────┼──────────────────────────────┐
│          │      Processing Layer     │                               │
│  ┌───────▼───────────────────────────▼───────┐                      │
│  │           XiangqiRecognizer               │                      │
│  │  ┌──────────────┐  ┌──────────────────┐   │                      │
│  │  │BoardDetector │  │  PieceDetector   │   │                      │
│  │  └──────┬───────┘  └────────┬─────────┘   │                      │
│  │         │                   │              │                      │
│  │  ┌──────▼───────────────────▼─────────┐   │                      │
│  │  │          FENGenerator              │   │                      │
│  │  └────────────────────────────────────┘   │                      │
│  └───────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
           │                           │
┌──────────┼───────────────────────────┼──────────────────────────────┐
│          │        ML Layer           │                               │
│  ┌───────▼───────┐          ┌────────▼────────┐                     │
│  │ YOLOv8-Seg    │          │    YOLOv8       │                     │
│  │ Board Model   │          │  Pieces Model   │                     │
│  └───────────────┘          └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Tech Stack

| Layer | Technology |
|-------|------------|
| ML Framework | PyTorch + Ultralytics YOLOv8 |
| Image Processing | OpenCV, NumPy |
| API Server | FastAPI + Uvicorn |
| Data Validation | Pydantic |
| Containerization | Docker |

---

## 2. Cấu Trúc Dự Án

```
xqrecognition/
│
├── config/                     # Configuration
│   ├── __init__.py
│   └── settings.py             # Global settings & constants
│
├── src/                        # Core source code
│   ├── __init__.py
│   ├── board_detector.py       # Board grid detection
│   ├── piece_detector.py       # Chess piece detection
│   ├── fen_generator.py        # FEN notation generation
│   └── pipeline.py             # Main recognition pipeline
│
├── scripts/                    # Utility scripts
│   ├── setup_data.py           # Dataset extraction
│   ├── train_board.py          # Board model training
│   ├── train_pieces.py         # Pieces model training
│   └── evaluate.py             # Evaluation script
│
├── models/                     # Trained models
│   ├── board_seg.pt            # Board segmentation model
│   └── pieces_det.pt           # Pieces detection model
│
├── data/                       # Datasets
│   ├── board_seg/              # Board segmentation data
│   └── pieces/                 # Pieces detection data
│
├── docs/                       # Documentation
│   ├── USAGE.md                # Usage guide
│   ├── ACCURACY_ANALYSIS.md    # Accuracy analysis
│   └── ARCHITECTURE.md         # This file
│
├── app.py                      # FastAPI server
├── train.py                    # Training CLI
├── detect.py                   # Detection CLI
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container config
└── README.md                   # Project overview
```

---

## 3. Luồng Xử Lý

### 3.1 Detection Pipeline Flow

```
                    Input Image
                         │
                         ▼
            ┌────────────────────────┐
            │    Image Validation    │
            │  - Format check        │
            │  - Size validation     │
            │  - Channel check       │
            └───────────┬────────────┘
                        │
           ┌────────────┴────────────┐
           │                         │
           ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │    Board     │          │    Piece     │
    │  Detection   │          │  Detection   │
    │  (Optional)  │          │  (Required)  │
    └──────┬───────┘          └──────┬───────┘
           │                         │
           │    ┌────────────────────┘
           │    │
           ▼    ▼
    ┌────────────────────────┐
    │    Grid Construction   │
    │  - ML-based (if board  │
    │    detection success)  │
    │  - Interpolation-based │
    │    (fallback)          │
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐
    │   Piece-to-Grid        │
    │      Mapping           │
    │  - Find nearest cell   │
    │  - Handle conflicts    │
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐
    │    FEN Generation      │
    │  - Build board matrix  │
    │  - Convert to string   │
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐
    │  Result Construction   │
    │  - FEN string          │
    │  - Piece details       │
    │  - Confidence score    │
    │  - Visualization       │
    └───────────┬────────────┘
                │
                ▼
           Output Result
```

### 3.2 Training Pipeline Flow

```
        Raw Dataset (ZIP)
               │
               ▼
    ┌────────────────────────┐
    │    Dataset Extraction  │
    │    (setup_data.py)     │
    └───────────┬────────────┘
               │
               ▼
    ┌────────────────────────┐
    │   Data Validation      │
    │  - Check structure     │
    │  - Verify labels       │
    │  - Check data.yaml     │
    └───────────┬────────────┘
               │
      ┌────────┴────────┐
      │                 │
      ▼                 ▼
┌───────────┐    ┌───────────┐
│  Board    │    │  Pieces   │
│ Training  │    │ Training  │
│(YOLOv8-seg│    │ (YOLOv8)  │
└─────┬─────┘    └─────┬─────┘
      │                │
      ▼                ▼
┌───────────┐    ┌───────────┐
│board_seg  │    │pieces_det │
│   .pt     │    │   .pt     │
└───────────┘    └───────────┘
```

---

## 4. Chi Tiết Các Module

### 4.1 BoardDetector (`src/board_detector.py`)

```python
class BoardDetector:
    """
    Phát hiện bàn cờ và xây dựng grid 9x10.

    Responsibilities:
    - Load và run YOLOv8-seg model
    - Detect 90 intersection points
    - Build grid từ detected points
    - Fallback methods khi detection thất bại
    """

    # Data structures
    @dataclass
    class Point:
        x: float
        y: float

    @dataclass
    class Grid:
        points: np.ndarray  # Shape: (10, 9, 2)
        cell_width: float
        cell_height: float

    # Main methods
    def detect_intersections(image) -> List[Point]
    def build_grid(intersections) -> Grid
    def build_grid_from_corners(corners) -> Grid  # Fallback
    def detect_board_corners(image) -> List[Point]  # CV fallback
    def visualize_grid(image, grid) -> np.ndarray
```

**Algorithm: Grid Construction**

```
1. Receive list of intersection points
2. Sort points by Y-coordinate (top to bottom)
3. Group into 10 rows based on Y-clustering
4. Within each row, sort by X-coordinate (left to right)
5. Verify each row has 9 points
6. If missing points, interpolate using neighbors
7. Calculate average cell dimensions
8. Return Grid object
```

### 4.2 PieceDetector (`src/piece_detector.py`)

```python
class PieceDetector:
    """
    Phát hiện và phân loại quân cờ.

    Responsibilities:
    - Load và run YOLOv8 detection model
    - Detect pieces với bounding boxes
    - Classify pieces (14 classes)
    - Apply NMS để loại bỏ duplicates
    """

    @dataclass
    class DetectedPiece:
        class_id: int
        class_name: str
        confidence: float
        bbox: Tuple[float, float, float, float]
        center: Tuple[float, float]
        fen_symbol: str

    # Main methods
    def detect_pieces(image) -> List[DetectedPiece]
    def non_max_suppression(pieces) -> List[DetectedPiece]
    def get_red_pieces(pieces) -> List[DetectedPiece]
    def get_black_pieces(pieces) -> List[DetectedPiece]
    def visualize_detections(image, pieces) -> np.ndarray
```

**Class Mapping:**

```python
PIECE_CLASSES = {
    0: ("Advisor_black", "a"),
    1: ("Advisor_red", "A"),
    2: ("Cannon_black", "c"),
    3: ("Cannon_red", "C"),
    4: ("Elephant_black", "b"),
    5: ("Elephant_red", "B"),
    6: ("General_black", "k"),
    7: ("General_red", "K"),
    8: ("Knight_black", "n"),
    9: ("Knight_red", "N"),
    10: ("Pawn_black", "p"),
    11: ("Pawn_red", "P"),
    12: ("Rook_black", "r"),
    13: ("Rook_red", "R"),
}
```

### 4.3 FENGenerator (`src/fen_generator.py`)

```python
class FENGenerator:
    """
    Tạo FEN notation từ board state.

    Responsibilities:
    - Map pieces to grid positions
    - Generate FEN string
    - Validate FEN against rules
    - Parse FEN back to board state
    """

    @dataclass
    class BoardState:
        board: List[List[Optional[str]]]  # 10x9 matrix
        pieces: List[Tuple[int, int, str]]  # (row, col, fen_symbol)

    # Main methods
    def map_pieces_to_grid(pieces, grid) -> BoardState
    def generate_fen(board_state) -> str
    def parse_fen(fen) -> BoardState
    def validate_fen(fen) -> Tuple[bool, List[str]]
    def compare_fen(fen1, fen2) -> Dict
```

**FEN Format:**

```
Row 0 (Black back): rnbakabnr
Row 1:              9          (empty)
Row 2:              1c5c1
Row 3:              p1p1p1p1p
Row 4:              9          (empty)
Row 5:              9          (empty)
Row 6:              P1P1P1P1P
Row 7:              1C5C1
Row 8:              9          (empty)
Row 9 (Red back):   RNBAKABNR

Full FEN: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
```

### 4.4 XiangqiRecognizer (`src/pipeline.py`)

```python
class XiangqiRecognizer:
    """
    Main pipeline kết hợp tất cả modules.

    Responsibilities:
    - Orchestrate detection pipeline
    - Handle errors và fallbacks
    - Generate final results
    """

    @dataclass
    class RecognitionResult:
        fen: str
        board_state: BoardState
        pieces: List[DetectedPiece]
        grid: Optional[Grid]
        image_shape: Tuple[int, int, int]
        confidence: float
        visualization: Optional[np.ndarray]
        errors: List[str]

    # Main methods
    def recognize(image_path) -> RecognitionResult
    def recognize_image(image_array) -> RecognitionResult
    def recognize_batch(image_paths) -> List[RecognitionResult]
```

---

## 5. Data Flow

### 5.1 Detection Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT                                   │
│  Image: np.ndarray (H, W, 3) BGR                                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BOARD DETECTION                               │
│  Input:  np.ndarray (H, W, 3)                                   │
│  Model:  YOLOv8-seg                                             │
│  Output: List[Point] (90 points with x, y coordinates)          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRID CONSTRUCTION                             │
│  Input:  List[Point]                                            │
│  Process: Sort, cluster, interpolate                            │
│  Output: Grid (10x9x2 array + cell dimensions)                  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PIECE DETECTION                               │
│  Input:  np.ndarray (H, W, 3)                                   │
│  Model:  YOLOv8                                                 │
│  Output: List[DetectedPiece] (class, bbox, center, confidence)  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRID MAPPING                                  │
│  Input:  List[DetectedPiece], Grid                              │
│  Process: Map each piece center to nearest grid cell            │
│  Output: BoardState (10x9 matrix with piece symbols)            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEN GENERATION                                │
│  Input:  BoardState                                             │
│  Process: Convert matrix to FEN string                          │
│  Output: str (e.g., "rnbakabnr/9/1c5c1/...")                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
│  RecognitionResult:                                             │
│  - fen: str                                                     │
│  - pieces: List[DetectedPiece]                                  │
│  - confidence: float                                            │
│  - visualization: Optional[np.ndarray]                          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Memory Considerations

```
Component               Typical Memory Usage
─────────────────────────────────────────────
Input Image (640x640)   ~1.2 MB
Board Model (YOLOv8n)   ~6 MB
Pieces Model (YOLOv8n)  ~6 MB
Inference (GPU)         ~500 MB
Inference (CPU)         ~200 MB
─────────────────────────────────────────────
Total (GPU)             ~520 MB
Total (CPU)             ~215 MB
```

---

## 6. API Design

### 6.1 Endpoints

```
GET  /              → API info
GET  /health        → Health check
POST /detect        → Detection (JSON response)
POST /detect/visualize → Detection (Image response)
POST /detect/json-with-image → Detection (JSON + base64 image)
```

### 6.2 Request/Response Models

```python
# Request: multipart/form-data with file

# Response: /detect
class DetectionResponse(BaseModel):
    fen: str
    pieces: List[PieceInfo]
    piece_count: int
    confidence: float
    errors: List[str]

class PieceInfo(BaseModel):
    class_id: int
    class_name: str
    display_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    center: List[float]  # [x, y]
    fen_symbol: str

# Response: /health
class HealthResponse(BaseModel):
    status: str
    board_model_loaded: bool
    pieces_model_loaded: bool
```

### 6.3 Error Handling

```python
# HTTP Status Codes
200 OK              - Successful detection
400 Bad Request     - Invalid image format
500 Internal Error  - Detection failure

# Error Response Format
{
    "detail": "Error message here"
}
```

---

## 7. Deployment

### 7.1 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
```

```bash
# Build và run
docker build -t xqrecognition .
docker run -p 8000:8000 xqrecognition
```

### 7.2 Production Considerations

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - WORKERS=4
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 7.3 Scaling Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  API #1  │    │  API #2  │    │  API #3  │
    │  (GPU)   │    │  (GPU)   │    │  (CPU)   │
    └──────────┘    └──────────┘    └──────────┘

# GPU instances: Heavy inference load
# CPU instances: Spillover handling
```

### 7.4 Monitoring

```python
# Recommended metrics to track
metrics = {
    "request_count": Counter,
    "request_latency": Histogram,
    "detection_confidence": Histogram,
    "piece_count": Histogram,
    "error_rate": Counter,
    "model_inference_time": Histogram,
}
```

---

## Appendix

### A. Configuration Options

```python
# config/settings.py

# Model paths
BOARD_SEG_MODEL = "models/board_seg.pt"
PIECES_DET_MODEL = "models/pieces_det.pt"

# Grid dimensions
GRID_COLS = 9
GRID_ROWS = 10

# Detection thresholds
BOARD_CONFIDENCE_THRESHOLD = 0.5
PIECE_CONFIDENCE_THRESHOLD = 0.5

# Training config
TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "patience": 20,
}
```

### B. Extension Points

```python
# Custom board detector
class CustomBoardDetector(BoardDetector):
    def detect_intersections(self, image):
        # Custom implementation
        pass

# Custom piece detector
class CustomPieceDetector(PieceDetector):
    def detect_pieces(self, image):
        # Custom implementation
        pass

# Use in pipeline
recognizer = XiangqiRecognizer()
recognizer.board_detector = CustomBoardDetector()
recognizer.piece_detector = CustomPieceDetector()
```
