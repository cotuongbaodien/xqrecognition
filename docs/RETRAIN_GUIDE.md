# Hướng Dẫn Retrain Model Pieces Detection

## Tại Sao Cần Retrain?

Model hiện tại (`pieces_det.pt`) được train trên dataset `pieces/` (519 ảnh app screenshot).
Khi test trên ảnh bàn cờ thật, accuracy gần 0% do domain gap (xem `DOMAIN_GAP_ANALYSIS.md`).

## Dataset Khuyến Nghị

### Tùy chọn 1: pieces_merged/ (Khuyến nghị)
```bash
# Đã có sẵn, 1431 ảnh train, bao gồm cả ảnh thật
python scripts/train_pieces.py --data data/pieces_merged/data.yaml
```

**Ưu điểm:**
- Đã có label sẵn, cùng 14 class names
- Bao gồm ảnh thật và ảnh app
- 1431 train + 203 valid + 30 test

### Tùy chọn 2: Kết hợp thêm xiangqi_pieces/ và xiangqi_v4/
Cần remap class names trước khi merge (dùng `scripts/merge_pieces_datasets.py`).

## Lệnh Retrain

### Cơ bản (YOLOv8s - khuyến nghị)
```bash
python scripts/train_pieces.py \
    --data data/pieces_merged/data.yaml \
    --pretrained yolov8s.pt \
    --epochs 150 \
    --batch-size 16 \
    --img-size 640
```

### Nếu có GPU yếu (giảm batch size)
```bash
python scripts/train_pieces.py \
    --data data/pieces_merged/data.yaml \
    --pretrained yolov8s.pt \
    --epochs 150 \
    --batch-size 8 \
    --img-size 640
```

### Nếu muốn nhanh hơn (YOLOv8n - ít chính xác hơn)
```bash
python scripts/train_pieces.py \
    --data data/pieces_merged/data.yaml \
    --pretrained yolov8n.pt \
    --epochs 100 \
    --batch-size 16
```

## Augmentation Đã Cấu Hình

Script train đã được cập nhật với các augmentation tối ưu cho bài toán cờ tướng:

| Augmentation | Giá trị | Lý do |
|-------------|---------|-------|
| `fliplr` | 0.5 | Lật ngang (bàn cờ đối xứng) |
| `flipud` | 0.0 | KHÔNG lật dọc (quân cờ có hướng) |
| `degrees` | ±15° | Xoay nhẹ (ảnh chụp nghiêng) |
| `perspective` | 0.0005 | Perspective nhẹ (góc chụp) |
| `hsv_h` | 0.02 | Biến đổi hue (bàn cờ khác màu) |
| `hsv_s` | 0.3 | Biến đổi saturation (ánh sáng) |
| `hsv_v` | 0.3 | Biến đổi brightness (bóng đổ) |
| `scale` | 0.3 | Scale variation |
| `mosaic` | 1.0 | Mosaic augmentation |

## Sau Khi Train

### 1. Model được lưu tự động
```
models/pieces_det.pt  ← Best model được copy tự động
runs/pieces_det/train/weights/best.pt  ← Bản gốc
runs/pieces_det/train/weights/last.pt  ← Checkpoint cuối
```

### 2. Đánh giá kết quả
```bash
# Đánh giá model mới trên test set
python scripts/evaluate.py pieces --model models/pieces_det.pt --data data/pieces_merged

# Đánh giá pipeline end-to-end trên fendata
python scripts/evaluate.py pipeline --test-dir data/fendata/images --ground-truth data/fendata/labels.csv
```

### 3. Kiểm tra confusion matrix
Xem file `runs/pieces_det/train/confusion_matrix.png` để phát hiện class nào vẫn bị nhầm lẫn.

## Yêu Cầu Phần Cứng

| Config | GPU VRAM | Thời gian ước tính |
|--------|----------|-------------------|
| YOLOv8n, batch=16 | 4GB | ~30 phút |
| YOLOv8s, batch=16 | 6GB | ~1 giờ |
| YOLOv8s, batch=8 | 4GB | ~1.5 giờ |
| YOLOv8m, batch=8 | 8GB | ~3 giờ |

## Tiếp Theo

Nếu accuracy vẫn chưa đạt 95% sau retrain:
1. Thêm data từ `prepare/` (786 ảnh có FEN, cần tạo YOLO labels)
2. Merge thêm `xiangqi_pieces/` và `xiangqi_v4/` (cần remap class)
3. Thử model lớn hơn (YOLOv8m)
4. Fine-tune thresholds trong `config/settings.py`
