# Hướng dẫn chuẩn bị dataset mới

## Class names chuẩn (18 classes)

Theo `data/itemdetection.yolov8.zip`. **TẤT CẢ data mới phải dùng đúng tên này**:

```yaml
nc: 18
names:
  0: black-advisor    # Sĩ đen
  1: black-cannon     # Pháo đen
  2: black-chariot    # Xe đen
  3: black-elephant   # Tượng đen
  4: black-general    # Tướng đen
  5: black-horse      # Mã đen
  6: black-soldier    # Tốt đen
  7: board-conner     # Góc bàn cờ (4 góc)
  8: palace-bottom    # Góc cung phía rìa bàn (cùng hàng góc bàn)
  9: palace-center    # Tâm cung (chỗ X giao nhau)
  10: palace-conner   # Góc cung phía trong bàn (sâu 2 hàng)
  11: red-advisor     # Sĩ đỏ
  12: red-cannon      # Pháo đỏ
  13: red-chariot     # Xe đỏ
  14: red-elephant    # Tượng đỏ
  15: red-general     # Tướng đỏ
  16: red-horse       # Mã đỏ
  17: red-soldier     # Tốt đỏ
```

## Các landmark cần label

### `board-conner` (4 điểm/bàn cờ)
- 4 góc của vùng bàn cờ thực tế (nơi 4 đường biên giao nhau)
- KHÔNG bao gồm phần viền/mép gỗ ngoài cùng

### `palace-bottom` (2 điểm/cung × 2 cung = 4 điểm/bàn cờ)
- Là 2 góc của cung tướng nằm ở **rìa bàn cờ** (cùng hàng với hàng tướng)
- Cung trên (đen): 2 góc ở row 0 (cùng hàng tướng đen)
- Cung dưới (đỏ): 2 góc ở row 9 (cùng hàng tướng đỏ)

### `palace-conner` (2 điểm/cung × 2 cung = 4 điểm/bàn cờ)
- Là 2 góc của cung tướng nằm ở **phía trong bàn cờ** (cách rìa 2 hàng)
- Cung trên: 2 góc ở row 2
- Cung dưới: 2 góc ở row 7

### `palace-center` (1 điểm/cung × 2 cung = 2 điểm/bàn cờ)
- Tâm của cung - nơi 2 đường chéo giao nhau
- Cung trên: row 1 col 4
- Cung dưới: row 8 col 4

## Tổng số label/ảnh đầy đủ

| Loại | Số instances |
|------|--------------|
| Pieces (đầu ván) | 32 quân (16 đỏ + 16 đen) |
| board-conner | 4 |
| palace-bottom | 4 (2 cho mỗi cung) |
| palace-conner | 4 (2 cho mỗi cung) |
| palace-center | 2 |
| **Tổng** | **46 instances/ảnh** |

## Tools để label

- **Roboflow** (khuyến nghị): https://app.roboflow.com - dễ dùng, export YOLO format
- **CVAT**: https://cvat.org - free, open source
- **LabelImg**: simple desktop tool

## Cấu trúc dataset

```
data/items/                      ← Chuẩn folder name
├── data.yaml                    ← Config file
├── train/
│   ├── images/                  ← .jpg, .png
│   └── labels/                  ← .txt YOLO format
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**data.yaml mẫu:**
```yaml
path: C:/Resources/xqrecognition/data/items
train: train/images
val: valid/images
test: test/images

nc: 18
names:
  0: black-advisor
  1: black-cannon
  ...
  17: red-soldier
```

## Yêu cầu chất lượng ảnh

| Tiêu chí | Yêu cầu |
|----------|---------|
| Resolution | ≥ 640x640 |
| Bàn cờ visibility | Toàn bộ bàn cờ trong khung hình |
| Bàn cờ chiếm ảnh | ≥ 50% diện tích |
| Góc chụp | Thẳng đứng hoặc nghiêng nhẹ (<15°) |
| Ánh sáng | Đều, không quá tối/sáng |
| Background | Có thể đa dạng (giúp model robust) |

## Tỷ lệ chia train/valid/test

- Train: 80%
- Valid: 15%
- Test: 5%

Hoặc dùng `random.seed(42)` để reproduce. Xem `scripts/` cho code split.

## Sau khi nhận data mới

1. Extract vào `data/items_v2/` (giữ `items/` cũ làm backup)
2. Verify class IDs đúng theo chuẩn 18 classes
3. Train: `python scripts/train_items.py --data data/items_v2/data.yaml --name items_v2 --seed 42`
4. Backup model: `cp models/items.pt models/backups/items_v2_<accuracy>.pt`
5. Test: `python scripts/evaluate_fendata.py`
6. Compare với baseline (items_v1: 69.2% mirror-tolerant)
