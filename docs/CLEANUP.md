# Kế Hoạch Tinh Gọn Code - Xiangqi Recognition System

## 1. Cấu Trúc Hiện Tại

### 1.1 Files trong `src/`

| File | Lines | Mô tả |
|------|-------|-------|
| `board_detector.py` | 736 | Board detection với nhiều phương pháp |
| `piece_detector.py` | 316 | Piece detection (cần giữ nguyên) |
| `fen_generator.py` | 760 | FEN generation với nhiều mapping methods |
| `pipeline.py` | 350 | Main pipeline |
| **Tổng** | **~2,162** | |

### 1.2 Classes trong `board_detector.py`

```
board_detector.py (736 lines)
├── Point (dataclass)           ~15 lines  ✅ GIỮ
├── Grid (dataclass)            ~30 lines  ✅ GIỮ
├── BoardDetector               ~290 lines
│   ├── detect_intersections()  ✅ GIỮ (backup fallback)
│   ├── build_grid()            ✅ GIỮ
│   ├── _interpolate_row()      ✅ GIỮ
│   ├── build_grid_from_corners() ❌ XÓA (không dùng)
│   ├── detect_board_corners()   ❌ XÓA (không dùng)
│   ├── _sort_corners()          ❌ XÓA (không dùng)
│   ├── build_grid_from_bbox()   ✅ GIỮ
│   └── visualize_grid()         ✅ GIỮ
├── BoardBoxDetector            ~90 lines  ✅ GIỮ NGUYÊN
└── CornerDetector              ~190 lines ❌ XÓA TOÀN BỘ
```

### 1.3 Methods trong `fen_generator.py`

```
fen_generator.py (760 lines)
├── BoardState (dataclass)                    ~40 lines  ✅ GIỮ
└── FENGenerator
    ├── map_pieces_to_grid()                  ~35 lines  ✅ GIỮ (primary)
    ├── map_pieces_to_grid_by_interpolation() ~45 lines  ✅ GIỮ (fallback)
    ├── map_pieces_to_grid_by_clustering()    ~60 lines  ❌ XÓA (không dùng)
    │   ├── _cluster_coordinates()            ~50 lines  ❌ XÓA
    │   └── _find_nearest_cluster()           ~15 lines  ❌ XÓA
    ├── map_pieces_to_grid_with_perspective() ~65 lines  ❌ XÓA (không dùng)
    │   ├── _find_row_positions_by_gaps()     ~55 lines  ❌ XÓA
    │   ├── _find_col_positions_by_gaps()     ~15 lines  ❌ XÓA
    │   ├── _find_nearest_row()               ~15 lines  ❌ XÓA
    │   └── _find_column_with_perspective()   ~35 lines  ❌ XÓA
    ├── generate_fen()                        ~30 lines  ✅ GIỮ
    ├── parse_fen()                           ~25 lines  ✅ GIỮ
    ├── validate_fen()                        ~35 lines  ✅ GIỮ
    ├── compare_fen()                         ~35 lines  ✅ GIỮ
    ├── get_starting_fen()                    ~5 lines   ✅ GIỮ
    ├── detect_board_orientation()            ~30 lines  ✅ GIỮ
    ├── flip_board()                          ~20 lines  ✅ GIỮ
    ├── normalize_board_orientation()         ~5 lines   ✅ GIỮ
    └── board_to_ascii()                      ~20 lines  ✅ GIỮ
```

### 1.4 Pipeline hiện tại (`pipeline.py`)

```python
# Hiện tại load 4 models:
- board_seg.pt      # BoardDetector (segmentation) - ít dùng
- board_det.pt      # BoardBoxDetector (detection) - CHÍNH
- board_corners.pt  # CornerDetector (pose) - KHÔNG DÙNG
- pieces_det.pt     # PieceDetector (detection) - CHÍNH
```

---

## 2. Phân Tích Code Thừa

### 2.1 CornerDetector (XÓA TOÀN BỘ)

**Lý do:**
- Dòng 190-192 trong `pipeline.py` comment: "Corner detection is disabled because the model was trained on different image types"
- Model được load nhưng không bao giờ được gọi
- ~190 lines code không sử dụng

### 2.2 BoardDetector methods không dùng

**Methods cần xóa:**
- `build_grid_from_corners()` - chỉ dùng với corner detection
- `detect_board_corners()` - CV-based fallback không dùng
- `_sort_corners()` - helper cho detect_board_corners

**Lý do:** Pipeline chỉ dùng `build_grid_from_bbox()` hoặc `build_grid()`

### 2.3 FENGenerator mapping methods thừa

**Methods cần xóa:**
- `map_pieces_to_grid_by_clustering()` và helpers
- `map_pieces_to_grid_with_perspective()` và helpers

**Lý do:** Pipeline chỉ dùng:
1. `map_pieces_to_grid()` - khi có grid
2. `map_pieces_to_grid_by_interpolation()` - fallback

---

## 3. Cấu Trúc Mới (Sau Tinh Gọn)

### 3.1 Dự kiến số dòng code

| File | Hiện tại | Sau tinh gọn | Giảm |
|------|----------|--------------|------|
| `board_detector.py` | 736 | ~350 | -386 (~52%) |
| `piece_detector.py` | 316 | 316 | 0 |
| `fen_generator.py` | 760 | ~450 | -310 (~41%) |
| `pipeline.py` | 350 | ~280 | -70 (~20%) |
| **Tổng** | **2,162** | **~1,396** | **-766 (~35%)** |

### 3.2 Models cần thiết

```
models/
├── board_det.pt     # BoardBoxDetector - CHÍNH
├── board_seg.pt     # BoardDetector - BACKUP (có thể xóa sau)
└── pieces_det.pt    # PieceDetector - CHÍNH
```

**Không cần:**
- `board_corners.pt` - Có thể xóa

### 3.3 Config cần cập nhật

```python
# config/settings.py

# XÓA:
CORNER_DET_MODEL = MODELS_DIR / "board_corners.pt"

# GIỮ:
BOARD_SEG_MODEL = MODELS_DIR / "board_seg.pt"  # backup
BOARD_DET_MODEL = MODELS_DIR / "board_det.pt"  # primary
PIECES_DET_MODEL = MODELS_DIR / "pieces_det.pt"
```

---

## 4. Pipeline Đơn Giản Hóa

### 4.1 Luồng xử lý mới

```
Input Image
     │
     ▼
┌─────────────────────────────────┐
│     1. Piece Detection          │
│     PieceDetector.detect_pieces │
│     + NMS                       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│     2. Board Detection          │
│     BoardBoxDetector.detect_board│
│     → build_grid_from_bbox()    │
└─────────────┬───────────────────┘
              │
              ├── Grid OK ──────────────────┐
              │                              │
              ▼                              ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│  3a. FALLBACK           │   │  3b. PRIMARY            │
│  Interpolation mapping  │   │  Grid-based mapping     │
│  (no board detection)   │   │  map_pieces_to_grid()   │
└─────────────┬───────────┘   └─────────────┬───────────┘
              │                              │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │  4. Normalize Orientation   │
              │  + Generate FEN             │
              └─────────────┬───────────────┘
                            │
                            ▼
                      Output FEN
```

### 4.2 Code mới cho pipeline

```python
class XiangqiRecognizer:
    def __init__(self, pieces_model, board_model=None, use_board_detection=True):
        self.piece_detector = PieceDetector(pieces_model)
        self.board_detector = BoardBoxDetector(board_model) if use_board_detection else None
        self.fen_generator = FENGenerator()

    def recognize(self, image):
        # 1. Detect pieces
        pieces = self.piece_detector.detect_pieces(image)
        pieces = self.piece_detector.non_max_suppression(pieces)

        # 2. Build grid (if board detection enabled)
        grid = None
        if self.board_detector:
            bbox = self.board_detector.detect_board(image)
            if bbox:
                grid = BoardDetector().build_grid_from_bbox(bbox)

        # 3. Map pieces to grid
        if grid:
            board_state = self.fen_generator.map_pieces_to_grid(pieces, grid)
        else:
            h, w = image.shape[:2]
            board_state = self.fen_generator.map_pieces_to_grid_by_interpolation(pieces, w, h)

        # 4. Normalize and generate FEN
        board_state = self.fen_generator.normalize_board_orientation(board_state)
        fen = self.fen_generator.generate_fen(board_state)

        return fen
```

---

## 5. Checklist Thực Hiện

### Phase 1: Backup
- [ ] Tạo branch mới: `refactor/cleanup-code`
- [ ] Commit code hiện tại

### Phase 2: Xóa code thừa
- [ ] `board_detector.py`: Xóa CornerDetector class
- [ ] `board_detector.py`: Xóa methods không dùng trong BoardDetector
- [ ] `fen_generator.py`: Xóa mapping methods không dùng
- [ ] `fen_generator.py`: Xóa helper methods liên quan
- [ ] `pipeline.py`: Xóa CornerDetector import và usage
- [ ] `config/settings.py`: Xóa CORNER_DET_MODEL

### Phase 3: Cập nhật
- [ ] Cập nhật imports trong các files
- [ ] Cập nhật docstrings
- [ ] Test functionality

### Phase 4: Documentation
- [ ] Cập nhật README.md
- [ ] Cập nhật ARCHITECTURE.md

---

## 6. Chuẩn Bị Cho Auto-Learn Pipeline

Sau khi tinh gọn, cấu trúc mới sẽ dễ dàng thêm auto-learn pipeline:

```
scripts/
├── auto_learn/
│   ├── __init__.py
│   ├── fen_parser.py       # Parse FEN → piece positions
│   ├── annotation_gen.py   # Generate YOLO annotations từ FEN
│   ├── dataset_builder.py  # Build dataset (train/val/test split)
│   └── trainer.py          # Training pipeline
```

**Workflow:**
1. Input: images/ + labels.csv (image_name, fen)
2. Detect board bbox → build grid
3. Parse FEN → get piece positions (row, col, type)
4. Generate bounding boxes tại các grid positions
5. Export YOLO format annotations
6. Train model với dataset mới
