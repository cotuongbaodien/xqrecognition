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
│  │  ┌──────────────────┐  ┌──────────────┐   │                      │
│  │  │ BoardBoxDetector │  │ PieceDetector│   │                      │
│  │  └────────┬─────────┘  └──────┬───────┘   │                      │
│  │           │                   │            │                      │
│  │  ┌────────▼───────────────────▼────────┐  │                      │
│  │  │          FENGenerator               │  │                      │
│  │  └─────────────────────────────────────┘  │                      │
│  └───────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
           │                           │
┌──────────┼───────────────────────────┼──────────────────────────────┐
│          │        ML Layer           │                               │
│  ┌───────▼───────┐          ┌────────▼────────┐                     │
│  │    YOLOv8     │          │    YOLOv8       │                     │
│  │  Board Model  │          │  Pieces Model   │                     │
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
│   └── settings.py             # Global settings & constants (~100 lines)
│
├── src/                        # Core source code (~1,400 lines total)
│   ├── __init__.py
│   ├── board_detector.py       # Board detection (~350 lines)
│   ├── piece_detector.py       # Piece detection (~320 lines)
│   ├── fen_generator.py        # FEN generation (~410 lines)
│   └── pipeline.py             # Main pipeline (~315 lines)
│
├── scripts/                    # Utility scripts
│   ├── setup_data.py           # Dataset extraction
│   ├── train_board.py          # Board model training
│   ├── train_pieces.py         # Pieces model training
│   └── evaluate.py             # Evaluation script
│
├── models/                     # Trained models
│   ├── board_det.pt            # Board detection (primary)
│   ├── board_seg.pt            # Board segmentation (fallback)
│   └── pieces_det.pt           # Pieces detection
│
├── data/                       # Datasets
│   ├── board_seg/              # Board segmentation data
│   └── pieces/                 # Pieces detection data
│
├── docs/                       # Documentation
│   ├── USAGE.md                # Usage guide
│   ├── ACCURACY_ANALYSIS.md    # Accuracy analysis
│   ├── CLEANUP.md              # Cleanup documentation
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
    │  - From bbox (primary) │
    │  - Interpolation       │
    │    (fallback)          │
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐
    │   Piece-to-Grid        │
    │      Mapping           │
    └───────────┬────────────┘
                │
                ▼
    ┌────────────────────────┐
    │  Normalize Orientation │
    │  + Generate FEN        │
    └───────────┬────────────┘
                │
                ▼
           Output FEN
```

---

## 4. Chi Tiết Các Module

### 4.1 BoardDetector (`src/board_detector.py`)

```python
@dataclass
class Point:
    x: float
    y: float

@dataclass
class Grid:
    points: np.ndarray  # Shape: (10, 9, 2)
    cell_width: float
    cell_height: float

class BoardDetector:
    """YOLOv8-seg based intersection detection (fallback)."""
    def detect_intersections(image) -> List[Point]
    def build_grid(intersections) -> Grid
    def build_grid_from_bbox(bbox) -> Grid
    def visualize_grid(image, grid) -> np.ndarray

class BoardBoxDetector:
    """YOLOv8 based board bounding box detection (primary)."""
    def detect_board(image) -> Tuple[x1, y1, x2, y2]
    def build_grid_from_detection(image) -> Grid
```

### 4.2 PieceDetector (`src/piece_detector.py`)

```python
@dataclass
class DetectedPiece:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    fen_symbol: str

class PieceDetector:
    def detect_pieces(image) -> List[DetectedPiece]
    def non_max_suppression(pieces) -> List[DetectedPiece]
    def get_red_pieces(pieces) -> List[DetectedPiece]
    def get_black_pieces(pieces) -> List[DetectedPiece]
    def visualize_detections(image, pieces) -> np.ndarray
```

### 4.3 FENGenerator (`src/fen_generator.py`)

```python
@dataclass
class BoardState:
    board: List[List[Optional[str]]]  # 10x9 matrix
    pieces: List[Tuple[int, int, str]]

class FENGenerator:
    def map_pieces_to_grid(pieces, grid) -> BoardState
    def map_pieces_to_grid_by_interpolation(pieces, w, h) -> BoardState
    def generate_fen(board_state) -> str
    def parse_fen(fen) -> BoardState
    def validate_fen(fen) -> Tuple[bool, List[str]]
    def compare_fen(fen1, fen2) -> Dict
    def normalize_board_orientation(board_state) -> BoardState
```

### 4.4 XiangqiRecognizer (`src/pipeline.py`)

```python
@dataclass
class RecognitionResult:
    fen: str
    board_state: BoardState
    pieces: List[DetectedPiece]
    grid: Optional[Grid]
    confidence: float
    visualization: Optional[np.ndarray]
    errors: List[str]

class XiangqiRecognizer:
    def recognize(image_path) -> RecognitionResult
    def recognize_image(image_array) -> RecognitionResult
    def recognize_batch(image_paths) -> List[RecognitionResult]
```

---

## 5. Data Flow

### 5.1 Detection Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Image (H, W, 3) BGR                                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PIECE DETECTION (YOLOv8)                                        │
│  Output: List[DetectedPiece]                                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  BOARD DETECTION (YOLOv8)                                        │
│  Output: bbox (x1, y1, x2, y2)                                  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  GRID CONSTRUCTION                                               │
│  Output: Grid (10x9x2 array)                                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  GRID MAPPING                                                    │
│  Output: BoardState (10x9 matrix)                               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEN GENERATION                                                  │
│  Output: "rnbakabnr/9/1c5c1/..."                                │
└─────────────────────────────────────────────────────────────────┘
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

### 6.2 Response Models

```python
class DetectionResponse(BaseModel):
    fen: str
    pieces: List[PieceInfo]
    piece_count: int
    confidence: float
    errors: List[str]
```

---

## 7. Configuration

### 7.1 Model Paths

```python
BOARD_DET_MODEL = "models/board_det.pt"   # Primary board detection
BOARD_SEG_MODEL = "models/board_seg.pt"   # Fallback segmentation
PIECES_DET_MODEL = "models/pieces_det.pt" # Pieces detection
```

### 7.2 Grid Dimensions

```python
GRID_COLS = 9   # Files a-i
GRID_ROWS = 10  # Ranks 0-9
TOTAL_INTERSECTIONS = 90
```

### 7.3 Piece Classes (14 total)

| ID | Name | FEN |
|----|------|-----|
| 0 | Advisor_black | a |
| 1 | Advisor_red | A |
| 2 | Cannon_black | c |
| 3 | Cannon_red | C |
| 4 | Elephant_black | b |
| 5 | Elephant_red | B |
| 6 | General_black | k |
| 7 | General_red | K |
| 8 | Knight_black | n |
| 9 | Knight_red | N |
| 10 | Pawn_black | p |
| 11 | Pawn_red | P |
| 12 | Rook_black | r |
| 13 | Rook_red | R |

---

## Appendix: Code Statistics

| File | Lines |
|------|-------|
| `board_detector.py` | ~350 |
| `piece_detector.py` | ~320 |
| `fen_generator.py` | ~410 |
| `pipeline.py` | ~315 |
| **Total src/** | **~1,400** |
