# Phân Tích Độ Chính Xác & Giải Pháp Cải Thiện

## Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Các Nguồn Lỗi](#2-các-nguồn-lỗi)
3. [Metrics Đánh Giá](#3-metrics-đánh-giá)
4. [Phân Tích Từng Module](#4-phân-tích-từng-module)
5. [Giải Pháp Cải Thiện](#5-giải-pháp-cải-thiện)
6. [Roadmap Phát Triển](#6-roadmap-phát-triển)

---

## 1. Tổng Quan Hệ Thống

### 1.1 Pipeline Xử Lý

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 1: Board Detection           │  ← Nguồn lỗi #1
│  - Detect 90 intersection points    │
│  - Build 9x10 grid                  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 2: Piece Detection           │  ← Nguồn lỗi #2
│  - Detect pieces (14 classes)       │
│  - Get bounding boxes               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 3: Piece-to-Grid Mapping     │  ← Nguồn lỗi #3
│  - Map piece centers to grid        │
│  - Handle edge cases                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 4: FEN Generation            │  ← Ít lỗi (logic đơn giản)
│  - Convert to FEN string            │
└─────────────────────────────────────┘
    │
    ▼
Output FEN
```

### 1.2 Tỷ Lệ Lỗi Dự Kiến

| Stage | Độ chính xác dự kiến | Ảnh hưởng đến kết quả |
|-------|---------------------|----------------------|
| Board Detection | 85-95% | Cao |
| Piece Detection | 90-98% | Rất cao |
| Grid Mapping | 95-99% | Trung bình |
| FEN Generation | ~100% | Thấp |
| **Tổng thể** | **75-90%** | - |

---

## 2. Các Nguồn Lỗi

### 2.1 Lỗi Từ Input Image

| Loại lỗi | Mô tả | Tần suất |
|----------|-------|----------|
| Góc chụp nghiêng | Ảnh chụp không thẳng đứng | Cao |
| Ánh sáng không đều | Bóng đổ, ngược sáng | Cao |
| Mờ/Noise | Ảnh rung, chất lượng thấp | Trung bình |
| Che khuất | Tay, vật thể che một phần | Thấp |
| Reflection | Phản chiếu trên bề mặt bóng | Trung bình |

### 2.2 Lỗi Board Detection

```
Vấn đề                          Impact    Giải pháp
─────────────────────────────────────────────────────────
Không detect đủ 90 điểm         Cao       Fallback interpolation
Detect sai vị trí điểm          Cao       Improve model/data
Grid bị méo                     Trung     Perspective correction
Background clutter              Trung     Better segmentation
```

### 2.3 Lỗi Piece Detection

| Vấn đề | Nguyên nhân | Ảnh hưởng |
|--------|-------------|-----------|
| Miss detection | Quân bị che, mờ | Thiếu quân trong FEN |
| False positive | Background giống quân cờ | Thêm quân sai |
| Wrong class | Quân tương tự (Sĩ vs Tướng) | FEN sai class |
| Duplicate detection | NMS không hiệu quả | 2 quân tại 1 vị trí |

### 2.4 Lỗi Grid Mapping

```python
# Vấn đề: Quân nằm giữa 2 ô
#
#     [ô A]     [ô B]
#         ●────●
#           ↑
#        quân cờ
#
# → Cần logic chọn ô gần nhất
```

---

## 3. Metrics Đánh Giá

### 3.1 Detection Metrics

```python
# Pieces Detection
mAP@50          # Mean Average Precision at IoU=0.5
mAP@50-95       # Mean AP averaged over IoU 0.5-0.95
Precision       # TP / (TP + FP)
Recall          # TP / (TP + FN)

# Per-class metrics
AP_per_class    # AP cho từng loại quân cờ
```

### 3.2 End-to-End Metrics

```python
# FEN Accuracy
FEN_exact_match     # Tỷ lệ FEN khớp hoàn toàn
FEN_position_acc    # Tỷ lệ vị trí quân đúng

# Piece-level
piece_count_acc     # Đếm đúng số quân
piece_position_acc  # Vị trí từng quân đúng
piece_class_acc     # Class từng quân đúng
```

### 3.3 Script Đánh Giá

```python
# scripts/detailed_evaluate.py

def evaluate_detailed(predictions, ground_truths):
    """
    Đánh giá chi tiết theo từng tiêu chí
    """
    results = {
        "total_images": len(predictions),
        "fen_exact_match": 0,
        "piece_detection": {
            "total_gt_pieces": 0,
            "total_pred_pieces": 0,
            "correct_pieces": 0,
            "missed_pieces": 0,
            "false_positives": 0,
        },
        "per_class_accuracy": {},
        "confusion_matrix": {},
    }

    for img_name, gt_fen in ground_truths.items():
        pred_fen = predictions.get(img_name, "")

        # FEN exact match
        if pred_fen == gt_fen:
            results["fen_exact_match"] += 1

        # Piece-level analysis
        gt_board = parse_fen(gt_fen)
        pred_board = parse_fen(pred_fen)

        # Compare each position
        for row in range(10):
            for col in range(9):
                gt_piece = gt_board[row][col]
                pred_piece = pred_board[row][col]

                # Update statistics...

    return results
```

---

## 4. Phân Tích Từng Module

### 4.1 Board Detection Analysis

**Điểm mạnh:**
- Segmentation model có thể xử lý nhiều góc độ
- Polygon output chính xác hơn bounding box

**Điểm yếu:**
- Cần đủ 90 điểm để build grid chính xác
- Sensitive với background clutter
- Không hoạt động tốt với ảnh nghiêng > 30°

**Cải thiện:**
```python
# Thêm confidence weighting
def build_grid_weighted(intersections, confidences):
    """
    Sử dụng confidence để weight các điểm
    khi interpolate grid
    """
    # High confidence points anchor the grid
    # Low confidence points are adjusted
    pass

# Thêm geometric validation
def validate_grid_geometry(grid):
    """
    Kiểm tra grid có đúng hình học không
    - Khoảng cách giữa các điểm đều nhau
    - Góc giữa các đường thẳng
    """
    pass
```

### 4.2 Piece Detection Analysis

**Điểm mạnh:**
- YOLO rất nhanh và accurate
- 14 classes đủ cover tất cả quân cờ

**Điểm yếu:**
- Khó phân biệt quân tương tự (Sĩ đỏ vs Tướng đỏ khi nhỏ)
- Sensitive với rotation
- May miss partially occluded pieces

**Confusion Matrix dự kiến:**

```
                 Predicted
              A   a   C   c   B   b   K   k   N   n   P   p   R   r
Actual    A [ H   -   -   -   -   -   M   -   -   -   -   -   -   - ]
          a [ -   H   -   -   -   -   -   M   -   -   -   -   -   - ]
          C [ -   -   H   -   -   -   -   -   -   -   L   -   L   - ]
          c [ -   -   -   H   -   -   -   -   -   -   -   L   -   L ]
          B [ -   -   -   -   H   -   -   -   -   -   -   -   -   - ]
          b [ -   -   -   -   -   H   -   -   -   -   -   -   -   - ]
          K [ M   -   -   -   -   -   H   -   -   -   -   -   -   - ]
          k [ -   M   -   -   -   -   -   H   -   -   -   -   -   - ]
          N [ -   -   -   -   -   -   -   -   H   -   -   -   -   - ]
          n [ -   -   -   -   -   -   -   -   -   H   -   -   -   - ]
          P [ -   -   L   -   -   -   -   -   -   -   H   -   -   - ]
          p [ -   -   -   L   -   -   -   -   -   -   -   H   -   - ]
          R [ -   -   L   -   -   -   -   -   -   -   -   -   H   - ]
          r [ -   -   -   L   -   -   -   -   -   -   -   -   -   H ]

H = High accuracy, M = Medium confusion, L = Low confusion
```

### 4.3 Grid Mapping Analysis

**Vấn đề chính:**
1. Piece center không chính xác khi bounding box lệch
2. Quân nằm ở ranh giới giữa 2 ô

**Giải pháp:**
```python
def improved_grid_mapping(piece_center, grid, piece_bbox):
    """
    Cải thiện mapping bằng cách:
    1. Adjust center based on bbox aspect ratio
    2. Use weighted distance with neighbor cells
    3. Apply game rules validation
    """
    cx, cy = piece_center
    x1, y1, x2, y2 = piece_bbox

    # Adjust for tall pieces (like Generals)
    bbox_height = y2 - y1
    bbox_width = x2 - x1

    if bbox_height > bbox_width * 1.2:
        # Piece is tall, center may be too high
        cy = y1 + bbox_height * 0.6  # Shift down

    # Find nearest cell with tolerance
    row, col = grid.get_nearest_cell(cx, cy)

    # Validate with game rules
    # (e.g., Generals can only be in palace)

    return row, col
```

---

## 5. Giải Pháp Cải Thiện

### 5.1 Cải Thiện Data

#### A. Data Augmentation

```python
# Augmentation cho training
augmentations = {
    # Geometric
    "rotation": (-30, 30),          # Xoay ±30°
    "perspective": 0.2,              # Perspective transform
    "scale": (0.8, 1.2),            # Scale ±20%

    # Photometric
    "brightness": (-0.3, 0.3),
    "contrast": (0.7, 1.3),
    "saturation": (0.7, 1.3),
    "hue": (-0.1, 0.1),

    # Noise
    "gaussian_noise": 0.02,
    "blur": (0, 3),

    # Occlusion simulation
    "cutout": {"num_holes": 3, "max_size": 30},
}
```

#### B. Synthetic Data Generation

```python
# Tạo dữ liệu tổng hợp
def generate_synthetic_board():
    """
    Tạo ảnh bàn cờ tổng hợp với:
    1. Random board background
    2. Random piece positions
    3. Random lighting/shadows
    4. Random camera angle
    """
    # Load board template
    board = load_random_board_template()

    # Generate random game state
    pieces = generate_valid_game_state()

    # Place pieces on board
    for piece, position in pieces:
        board = place_piece(board, piece, position)

    # Apply augmentations
    board = apply_augmentations(board)

    return board, generate_labels(pieces)
```

### 5.2 Cải Thiện Model

#### A. Model Architecture

```python
# Sử dụng model lớn hơn cho accuracy
model_options = {
    "fast": "yolov8n.pt",      # Nano - nhanh nhất
    "balanced": "yolov8s.pt",   # Small - cân bằng
    "accurate": "yolov8m.pt",   # Medium - chính xác hơn
    "best": "yolov8l.pt",       # Large - chính xác nhất
}

# Ensemble multiple models
class EnsembleDetector:
    def __init__(self):
        self.models = [
            YOLO("yolov8s.pt"),
            YOLO("yolov8m.pt"),
        ]

    def detect(self, image):
        all_predictions = []
        for model in self.models:
            preds = model(image)
            all_predictions.append(preds)

        # Merge predictions with weighted voting
        return merge_predictions(all_predictions)
```

#### B. Multi-task Learning

```python
# Train single model for both tasks
# (detection + classification refinement)

class MultiTaskModel:
    """
    Stage 1: Detect all chess pieces (binary)
    Stage 2: Classify detected pieces (14 classes)
    """
    def __init__(self):
        self.detector = YOLO("yolov8s.pt")  # Detection
        self.classifier = CNN()              # Fine-grained classification

    def forward(self, image):
        # Detect all pieces
        detections = self.detector(image)

        # Classify each detection
        for det in detections:
            crop = extract_crop(image, det.bbox)
            det.class_id = self.classifier(crop)

        return detections
```

### 5.3 Cải Thiện Post-Processing

#### A. Game Rules Validation

```python
class XiangqiRulesValidator:
    """
    Validate FEN against Xiangqi rules
    """
    RULES = {
        # Piece count limits
        "max_pieces": {
            'k': 1, 'K': 1,  # 1 General each
            'a': 2, 'A': 2,  # 2 Advisors each
            'b': 2, 'B': 2,  # 2 Elephants each
            'n': 2, 'N': 2,  # 2 Knights each
            'r': 2, 'R': 2,  # 2 Rooks each
            'c': 2, 'C': 2,  # 2 Cannons each
            'p': 5, 'P': 5,  # 5 Pawns each
        },

        # Position constraints
        "general_positions": {
            'K': [(7,3), (7,4), (7,5), (8,3), (8,4), (8,5), (9,3), (9,4), (9,5)],
            'k': [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)],
        },

        "advisor_positions": {
            'A': [(7,3), (7,5), (8,4), (9,3), (9,5)],
            'a': [(0,3), (0,5), (1,4), (2,3), (2,5)],
        },

        "elephant_positions": {
            'B': [(7,0), (7,2), (7,4), (7,6), (7,8), (9,0), (9,2), (9,4), (9,6), (9,8)],
            'b': [(0,0), (0,2), (0,4), (0,6), (0,8), (2,0), (2,2), (2,4), (2,6), (2,8)],
        },
    }

    def validate_and_correct(self, board_state, detections):
        """
        Validate board state and attempt to correct errors
        """
        corrections = []

        # Check piece counts
        counts = board_state.count_pieces()
        for piece, max_count in self.RULES["max_pieces"].items():
            if counts.get(piece, 0) > max_count:
                # Remove lowest confidence duplicates
                corrections.append(f"Remove extra {piece}")

        # Check position constraints
        for row in range(10):
            for col in range(9):
                piece = board_state.get_piece(row, col)
                if piece:
                    valid_positions = self.RULES.get(f"{piece}_positions", {}).get(piece)
                    if valid_positions and (row, col) not in valid_positions:
                        # Piece in invalid position
                        corrections.append(f"Invalid position for {piece} at ({row},{col})")

        return corrections
```

#### B. Confidence-based Filtering

```python
def filter_low_confidence_detections(detections, threshold=0.5):
    """
    Filter out low confidence detections
    with adaptive threshold based on detection density
    """
    if len(detections) <= 32:
        # Normal game - keep all above threshold
        return [d for d in detections if d.confidence >= threshold]
    else:
        # Too many detections - increase threshold
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        return sorted_dets[:32]  # Keep top 32
```

### 5.4 Cải Thiện Board Detection

#### A. Hybrid Approach

```python
class HybridBoardDetector:
    """
    Combine multiple methods for robust board detection
    """
    def __init__(self):
        self.seg_model = YOLO("board_seg.pt")  # ML-based
        self.cv_detector = CVBoardDetector()    # Traditional CV

    def detect(self, image):
        # Try ML-based first
        ml_result = self.seg_model(image)

        if self._is_valid_detection(ml_result):
            return ml_result

        # Fallback to traditional CV
        cv_result = self.cv_detector.detect(image)

        if self._is_valid_detection(cv_result):
            return cv_result

        # Last resort: assume standard board position
        return self._interpolate_grid(image)

    def _is_valid_detection(self, result):
        # Check if detection has enough points
        # and forms a valid grid
        return len(result.points) >= 70  # At least 70/90 points
```

#### B. Traditional CV Methods

```python
class CVBoardDetector:
    """
    Traditional computer vision approach for board detection
    """
    def detect(self, image):
        # 1. Find board contour
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 2. Find largest quadrilateral
        board_contour = self._find_board_contour(contours)

        # 3. Perspective transform
        warped = self._warp_board(image, board_contour)

        # 4. Detect grid lines
        grid = self._detect_grid_lines(warped)

        return grid

    def _detect_grid_lines(self, image):
        """Detect grid lines using Hough transform"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect lines
        lines = cv2.HoughLinesP(gray, 1, np.pi/180, 100)

        # Cluster into horizontal and vertical
        h_lines = self._cluster_horizontal(lines)
        v_lines = self._cluster_vertical(lines)

        # Find intersections
        intersections = self._find_intersections(h_lines, v_lines)

        return intersections
```

---

## 6. Roadmap Phát Triển

### Phase 1: Baseline (Hiện tại)
- [x] Basic pipeline implementation
- [x] YOLOv8 for piece detection
- [x] Simple grid mapping
- [ ] Target accuracy: **75-80%** FEN exact match

### Phase 2: Improved Accuracy
- [ ] Data augmentation pipeline
- [ ] Larger YOLO model (yolov8m)
- [ ] Game rules validation
- [ ] Hybrid board detection
- [ ] Target accuracy: **85-90%** FEN exact match

### Phase 3: Production Ready
- [ ] Ensemble models
- [ ] Confidence calibration
- [ ] Error correction with game rules
- [ ] Multi-angle support
- [ ] Target accuracy: **92-95%** FEN exact match

### Phase 4: Advanced Features
- [ ] Real-time video processing
- [ ] Move detection between frames
- [ ] Game analysis integration
- [ ] Mobile optimization
- [ ] Target: **95%+** accuracy with <100ms latency

---

## Kết Luận

### Ưu Tiên Cải Thiện (theo ROI)

| Cải thiện | Effort | Impact | Priority |
|-----------|--------|--------|----------|
| Data augmentation | Low | High | **1** |
| Game rules validation | Low | Medium | **2** |
| Larger model (yolov8m) | Low | Medium | **3** |
| Hybrid board detection | Medium | High | **4** |
| Ensemble models | High | Medium | 5 |
| Synthetic data | High | High | 6 |

### Metrics Mục Tiêu

```
Current  →  Phase 2  →  Phase 3
  75%         88%         95%
```

### Resources Cần Thiết

- **Training**: GPU với 8GB+ VRAM
- **Dataset**: 5000+ labeled images
- **Time**: 2-4 tuần cho Phase 2
