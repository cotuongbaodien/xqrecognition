# Kiểm Tra Chất Lượng Data

## Tổng Quan Dataset

### pieces_merged/ (Dataset Training Chính)
| Nguồn | Train | Valid | Test | Loại ảnh |
|-------|-------|-------|------|----------|
| pieces/ (old_) | 519 | 50 | 25 | App screenshot |
| xiangqi_pieces/ (new_) | 279 | 39 | 5 | Ảnh thật (bàn cờ thật) |
| xiangqi_v4/ (v4_) | 633 | 114 | 0 | Hỗn hợp |
| fendata/ (fen_) | 51 | 0 | 0 | Ảnh thật (auto-labeled) |
| prepare/ (prep_) | 51 | 0 | 0 | Ảnh thật (auto-labeled) |
| **Tổng** | **1533** | **203** | **30** | |

### Cảnh báo: xiangqi_pieces Mapping
Dataset `xiangqi_pieces` có tên class khác biệt:
- Có "chariot" và "rook" nhưng KHÔNG có "cannon"
- Mapping hiện tại: chariot → Cannon, rook → Rook
- **Cần verify**: mapping có thể sai (chariot nghĩa là 車 = Rook, không phải Cannon)
- Nếu sai, ~323 ảnh trong pieces_merged bị label ngược Cannon/Rook

### Data prepare/ - Chất lượng thấp
- 786 ảnh nhưng chỉ 52 có FEN label (734 trống)
- Board detection fail trên 7/52 ảnh có label
- Chỉ thêm được 51 ảnh auto-labeled

### Data fendata/ - Chất lượng trung bình
- 99 ảnh, tất cả có FEN label
- Board detection fail trên 48/99 ảnh (confidence=0.3)
- Chỉ thêm được 51 ảnh auto-labeled
- FEN label đầu tiên (001.png) có format lạ: `02p6` (chứa số 0)

## Vấn Đề Chất Lượng Data

### 1. Auto-labeled data có thể có noise
Ảnh từ fendata/ và prepare/ được label tự động bằng:
- board_det.pt detect bounding box
- FEN ground truth → tính vị trí quân

Nếu bbox hơi lệch → tọa độ quân sai → training bị noise

### 2. Class imbalance
Từ dataset pieces/ (nguồn chính):
- Pawn: ~2000 labels (nhiều nhất)
- General: ~500 labels (ít nhất)
- Tỉ lệ 4:1 → model có thể yếu trên class ít

### 3. Domain imbalance
- ~1152 ảnh app screenshot (pieces/ + phần xiangqi_v4/)
- ~381 ảnh thật (xiangqi_pieces/ + fen_/prep_)
- Tỉ lệ 3:1 app:thật → model vẫn bias về app

## Khuyến Nghị

1. **Verify xiangqi_pieces mapping**: Kiểm tra vài ảnh xem chariot = Cannon hay Rook
2. **Thêm data ảnh thật**: Label thêm FEN cho 734 ảnh prepare/ trống
3. **Tăng augmentation**: Đặc biệt cho color jitter (đỏ/đen) và perspective
4. **Validation set cần đa dạng**: Hiện tại valid/ chỉ từ pieces/ và xiangqi_pieces/, chưa có ảnh auto-labeled
