# Kế Hoạch Cải Thiện Độ Chính Xác FEN

## Mục Tiêu
- Hiện tại: 75-85% FEN exact match (trên ảnh app), ~0% trên ảnh thật
- Mục tiêu: 95% FEN exact match trên cả ảnh app và ảnh thật

## Phân Tích Vấn Đề Chính

### 1. Domain Gap - Vấn đề lớn nhất

**Phát hiện:** Model `pieces_det.pt` được train trên dataset `pieces/` (519 ảnh) - chỉ gồm ảnh screenshot từ app cờ online. Khi test trên ảnh chụp bàn cờ thật (fendata/), model phân loại sai nặng:
- Detect 28 quân nhưng phân loại sai (vd: 8 Elephant_red thay vì max 2)
- Cannon, Pawn, Rook bị nhầm thành Elephant
- Accuracy ~0% trên fendata

**Nguyên nhân:** Quân cờ trong app (màu vàng/nâu, có hiệu ứng) khác hoàn toàn với quân cờ thật (màu trắng, chữ Hán đơn giản).

**Bảng kiểm tra:**

| Dataset | Loại ảnh | Số ảnh | Accuracy |
|---------|----------|--------|----------|
| test/ (7 ảnh) | Screenshot app | 7 | ~95% (32/32 quân, conf>0.95) |
| fendata/ (99 ảnh) | Ảnh chụp thật | 99 | ~0% (sai class nặng) |

### 2. Pipeline Code Issues (đã fix)

| Vấn đề | File | Fix |
|--------|------|-----|
| Collision: 2 quân map cùng 1 ô, mất quân thứ 2 | fen_generator.py | Chọn quân có confidence cao nhất |
| NMS IoU=0.5 quá cao, loại quân sát nhau | pipeline.py | Giảm xuống 0.35 |
| Không validate theo luật cờ | pipeline.py | Thêm RulesValidator |
| Board grid không có margin | board_detector.py | Thêm margin 2% |
| Orientation dựa trên avg row (yếu) | fen_generator.py | Dùng vị trí Tướng (K/k) |
| Fallback intersection threshold=20 (quá thấp) | pipeline.py | Tăng lên 50 |

### 3. Model Architecture

- Hiện tại: YOLOv8 **Nano** (yolov8n) - 3.2M params
- Đề xuất: YOLOv8 **Small** (yolov8s) - 11.2M params (x3.5 capacity)
- Lý do: Model nano không đủ capacity phân biệt 14 class cờ tướng với nhiều style khác nhau

## Kế Hoạch Thực Hiện

### Phase 1: Pipeline Code Improvements [ĐÃ HOÀN THÀNH]

1. **Fix Grid Mapping** - fen_generator.py
   - Sort pieces theo confidence trước khi map
   - Collision resolution: giữ piece confidence cao nhất
   - Distance threshold: loại false positive xa grid cell > 0.6 * cell_size

2. **Giảm NMS threshold** - pipeline.py
   - IoU 0.5 -> 0.35
   - Cap max 32 pieces

3. **Game Rules Validator** - rules_validator.py (mới)
   - Validate số lượng quân (max per type)
   - Validate vị trí (tướng/sĩ trong cung, tượng nửa sân)
   - Auto-correct: loại quân thừa theo confidence

4. **Board Detection** - board_detector.py
   - Margin 2% cho build_grid_from_bbox
   - Tăng intersection threshold 20 -> 50

5. **Orientation Detection** - fen_generator.py
   - Dùng vị trí General (K/k) thay vì average row
   - Fallback: average row nếu không tìm thấy General

### Phase 2: Retrain Model [ĐANG THỰC HIỆN]

1. **Dataset:** Dùng `pieces_merged/` (1431 train, 203 valid) - đã có ảnh thật
2. **Model:** YOLOv8s thay vì YOLOv8n
3. **Augmentation:** Thêm perspective, rotation, lighting variations
4. **Script:** `scripts/train_pieces.py` - cập nhật config

### Phase 3: Evaluation & Iteration

1. Test trên fendata/ (99 ảnh có ground truth)
2. Test trên test/ (7 ảnh app)
3. Phân tích confusion matrix
4. Fine-tune thresholds

## Datasets Hiện Có

| Dataset | Số ảnh | Classes | Loại ảnh | Dùng cho |
|---------|--------|---------|----------|----------|
| pieces/ | 594 | 14 (standard) | App screenshot | Train hiện tại |
| pieces_merged/ | 1664 | 14 (standard) | App + ảnh thật | **Retrain** |
| xiangqi_pieces/ | 323 | 14 (tên khác) | Hỗn hợp | Cần remap class |
| xiangqi_v4/ | 747 | 14 (tên khác) | Hỗn hợp | Cần remap class |
| pieces_v2/ | ? | 8 (khác) | Hỗn hợp | Không tương thích |
| fendata/ | 99 (có FEN) | N/A | Ảnh thật | **Evaluation** |
| prepare/ | 786 (có FEN) | N/A | Mix | **Evaluation** |

## Kết Quả Dự Kiến

| Giai đoạn | FEN Accuracy (app) | FEN Accuracy (thật) |
|-----------|--------------------|---------------------|
| Trước fix | ~80% | ~0% |
| Sau Phase 1 (code fix) | ~85% | ~0% (model vẫn yếu) |
| Sau Phase 2 (retrain) | ~95% | ~85-90% |
| Sau Phase 3 (fine-tune) | ~97% | ~93-95% |
