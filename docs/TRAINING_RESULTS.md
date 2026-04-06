# Kết Quả Training & Đánh Giá

## Training Lần 1: YOLOv8s trên pieces_merged

### Cấu hình
- **Model:** YOLOv8s (11.2M params) - thay vì YOLOv8n cũ (3.2M params)
- **Dataset:** pieces_merged/ (1431 train, 203 valid)
- **GPU:** RTX 3060 12GB
- **Epochs:** 150 (early stopping patience=20)
- **Augmentation:** fliplr=0.5, degrees=15, perspective=0.0005, hsv_s=0.3, hsv_v=0.3

### Kết quả trên Validation Set
| Metric | Giá trị |
|--------|---------|
| mAP@50 | **0.994** (99.4%) |
| mAP@50-95 | **0.963** (96.3%) |
| Precision | 0.988 |
| Recall | 0.997 |

### So sánh với model cũ
| Metric | YOLOv8n (cũ) | YOLOv8s (mới) |
|--------|-------------|--------------|
| mAP@50 | ~0.90 | 0.994 |
| mAP@50-95 | ~0.80 | 0.963 |
| Model size | 6.3MB | ~22MB |

### Đánh giá trên fendata/ (ảnh thật)
| Metric | YOLOv8n (cũ) | YOLOv8s (mới) |
|--------|-------------|--------------|
| FEN exact match | 0/20 = 0% | 0/52 = 0% |
| Cell-level accuracy | ~5% | **50-76%** |
| Tốt nhất | N/A | 76% (image 050) |

### Phân tích lỗi còn lại

#### 1. Piece Classification - Đã cải thiện đáng kể
- Model cũ: 8 Elephant thay vì max 2 (confusion nặng)
- Model mới: Phân loại đúng hầu hết quân, confidence 0.85-0.95

#### 2. Grid Alignment - Vấn đề chính hiện tại
- Board bbox detect hơi lệch so với grid thực tế
- Margin 2% có thể chưa đủ hoặc quá nhiều tùy ảnh
- Quân ở rìa bàn bị map sai ô (shift 1 cột/hàng)

#### 3. Board Orientation - Vẫn có lỗi
- Một số ảnh bị flip sai hướng
- Dẫn đến toàn bộ FEN bị đảo ngược

#### 4. Pieces bị miss
- Một số quân confidence < 0.5 bị bỏ qua
- Đặc biệt quân nhỏ hoặc bị che khuất

## Các bước tiếp theo để đạt 95%

### Ưu tiên 1: Fix Grid Alignment
- Thử nhiều giá trị margin (0%, 1%, 2%, 3%, 5%)
- Dùng piece positions để validate/adjust grid
- Nếu detected pieces không khớp grid → adjust bbox

### Ưu tiên 2: Retrain với thêm data
- Hiện đã thêm 102 ảnh thật (fen_ + prep_) vào pieces_merged
- Tổng train: 1533 ảnh
- Chạy lại: `python scripts/train_pieces.py --data data/pieces_merged/data.yaml --pretrained yolov8s.pt --epochs 150 --batch-size 16 --device cuda`

### Ưu tiên 3: Giảm confidence threshold
- Thử piece confidence 0.3 thay vì 0.5
- Nhiều quân cờ thật có confidence 0.3-0.5

### Ưu tiên 4: Verify xiangqi_pieces mapping
- Kiểm tra chariot = Cannon hay Rook
- Nếu sai → re-merge → retrain

## Lệnh chạy retrain lần 2 (với data mới)

```bash
python scripts/train_pieces.py \
    --data data/pieces_merged/data.yaml \
    --pretrained yolov8s.pt \
    --epochs 150 \
    --batch-size 16 \
    --device cuda
```
