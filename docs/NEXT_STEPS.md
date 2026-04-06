# Các Bước Tiếp Theo

## Trạng thái hiện tại (2026-04-05)

### Đã hoàn thành
- [x] Pipeline code improvements (NMS, collision, rules validator, orientation)
- [x] Retrain model YOLOv8s trên pieces_merged (1431 ảnh)
- [x] Thêm 102 ảnh thật vào pieces_merged (fendata + prepare)
- [x] Tạo script sinh YOLO labels từ FEN (`scripts/generate_yolo_labels_from_fen.py`)
- [x] Documentation: IMPROVEMENT_PLAN, DOMAIN_GAP_ANALYSIS, DATA_AUDIT, TRAINING_RESULTS

### Kết quả hiện tại
- mAP@50 = 99.4% trên validation set
- FEN cell-level accuracy: 50-76% trên ảnh thật (tăng từ ~0%)
- FEN exact match: 0% (cần fix grid alignment)

## Việc cần làm (theo thứ tự ưu tiên)

### 1. Fix Grid Alignment (Impact cao nhất)
**Vấn đề:** Board bbox hơi lệch → quân map sai ô → FEN sai
**Cách làm:**
- Thử adaptive margin: dùng piece positions để calibrate grid
- So sánh detected piece centers với nearest grid cell → nếu offset hệ thống → adjust bbox
- File cần sửa: `src/board_detector.py`, `src/fen_generator.py`

### 2. Retrain lần 2 với thêm data
**Vấn đề:** Data hiện tại thiếu ảnh thật
**Cách làm:**
```bash
python scripts/train_pieces.py \
    --data data/pieces_merged/data.yaml \
    --pretrained yolov8s.pt \
    --epochs 150 \
    --batch-size 16 \
    --device cuda
```

### 3. Verify xiangqi_pieces class mapping
**Vấn đề:** Mapping chariot→Cannon có thể sai (chariot = 車 = Rook trong tiếng Trung)
**Cách làm:**
- Mở vài ảnh xiangqi_pieces, kiểm tra label class 2 (chariot) vs class 6 (rook)
- Nếu sai → chạy lại merge → retrain

### 4. Label thêm FEN cho prepare/
**Vấn đề:** 734/786 ảnh prepare/ chưa có FEN label
**Cách làm:** Dùng app/tool để label FEN cho các ảnh, sau đó:
```bash
python scripts/generate_yolo_labels_from_fen.py \
    --input data/prepare \
    --output data/pieces_merged \
    --split train \
    --prefix prep_ \
    --confidence 0.1
```

### 5. Giảm piece confidence threshold
**Vấn đề:** Nhiều quân cờ thật có confidence 0.3-0.5
**Cách làm:** Sửa `config/settings.py`: `PIECE_CONFIDENCE_THRESHOLD = 0.3`

## Mục tiêu
- Cell-level accuracy ≥ 95% trên fendata
- FEN exact match ≥ 90% trên fendata
