# Model Backups

Lưu các model đã train và đã verify accuracy. **KHÔNG ĐƯỢC XÓA** các file này.

## Models hiện có

| File | mAP50 | FEN exact (52 ảnh) | Mirror-tolerant | Mô tả |
|------|-------|--------------------|----------------|-------|
| `items_v1_mAP0.926.pt` | 0.926 | - | - | Items v1 (18 classes), train trên 122 ảnh itemdetection |
| `items_v1_original.pt` | 0.926 | - | - | Bản gốc giống items_v1, lưu thêm backup |
| `pieces_v1_67pct.pt` | 0.99 | 67.3% | 71.2% | Pieces detection (14 classes legacy), seed=42 |

## Cách restore

```bash
# Khôi phục pieces model
cp models/backups/pieces_v1_67pct.pt models/pieces_det.pt

# Khôi phục items model  
cp models/backups/items_v1_mAP0.926.pt models/items.pt
```

## Quy tắc backup

1. Mỗi lần train mới và verify accuracy → backup ngay với tên rõ ràng
2. Tên file format: `<modelname>_<version>_<metric>.pt`
3. Không bao giờ overwrite backup hiện có
