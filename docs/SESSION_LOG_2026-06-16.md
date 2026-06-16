# Nhật ký phiên làm việc — 2026-06-16

Chủ đề: thêm cửu cung (palace) vào board segmentation để fix lỗi chiều cam-dọc + cut-off,
audit toàn bộ giải thuật, và deploy `board_seg_v5`.

## 1. Dữ liệu & huấn luyện
- Dataset `download/board_segv3.yolov11.zip` = **dataset board cũ + vẽ chồng polygon palace** (không đổi ảnh board). 2 class `xiangqi-board`(0) + `xiangqi-palace`(1), polygon 4 đỉnh, mỗi ảnh 1 board + 2 palace. Tổng 153 ảnh (train 123 / valid 20 / test 10).
- Giải nén → `data/board_seg_v5/`, viết lại `data.yaml` đường dẫn tuyệt đối (yaml Roboflow dùng `../` resolve sai).
- Train YOLO11n-seg, imgsz 960, 200 epoch (early-stop @169, best @129, ~0.54h, RTX 3060). **val mAP50: board 0.995 / palace 0.995.**
- Lưu `models/backups/board_seg_v5.pt` (`scripts/train_board.py` cũ bị stale — import `config.settings` không tồn tại — nên train trực tiếp qua ultralytics).

## 2. Audit toàn giải thuật (multi-agent)
- Workflow đa-agent đối kháng: 7 chiều, 53 agent, **33 finding đã verify** → 13 fix ưu tiên.
- Báo cáo: `docs/ALGORITHM_AUDIT.md` (tiếng Việt).
- Phát hiện đắt giá nhất: seg-grid là đường chính nhưng **không sanity-check**; `_grid_from_4_corners` bỏ qua `image_shape`; quad sai-không-None chặn fallback (#1).

## 3. Các thay đổi code

### `boarddetection/board_segmenter.py` (viết lại)
- Hỗ trợ 2-class: lọc mask theo class (board 0 / palace 1); tương thích ngược model 1-class.
- `BoardSegResult`: thêm `palace_centers`, `palace_quads`, `clipped`.
- **Fix chiều 90° bằng trục palace→palace** (thay heuristic tỉ-lệ-cạnh): trục nối tâm 2 cửu cung = trục rank.
- `_order_quad_angular`: sắp góc theo `atan2` khi x±y bị tie (tilt ~45°).
- **Fix `_reduce_to_quad`** (bug nặng): approxPolyDP nhảy qua 4 (6→3) trên mask nhiễu → rơi `minAreaRect` → chữ nhật axis-aligned **mất phối cảnh**. Sửa: `_drop_to_4` bỏ đỉnh ít quan trọng nhất tới khi còn 4 (giữ hình thang).
- **Pin `detect(imgsz=640)`** — xem mục 4.

### `boarddetection/item_detector.py`
- `_grid_from_4_corners`: thêm guard `image_shape` (cell<5px, |w|<1e-6, >15% điểm ngoài rect mở rộng → None) (#1).
- `build_grid_from_quad`: thêm param `palace_quads`, `corners_reliable`. **Palace+board mutual-verify** qua `_grid_from_correspondences(ransac=True)` — board ghim ngoài, 8 góc cửu cung ghim trong; **chỉ refit khi cut-off** (`corners_reliable=False`), clean board giữ grid 4-góc thuần (refit bật-mọi-bàn từng warp/flip 13,32,48).
- `_assign_to_targets`: gán điểm palace vào (col,row) đã biết qua base grid (greedy, ngưỡng 0.6 ô).
- `_grid_from_correspondences`: thêm `ransac` (cv2.RANSAC, thresh 5px) loại outlier.

### `boarddetection/pipeline.py`
- Dùng `board_segmenter.detect()`; bơm 2 palace seg (wrap `Landmark`) vào orientation; truyền `palace_quads`, `corners_reliable=not seg.any_clipped`; cờ "board cut off".

## 4. ROOT CAUSE regression (quan trọng)
- Triệu chứng: deploy v5 → 86-set tụt 59→48..55, regress nhiều bàn clean (lệch ~1 ô / back-rank).
- Cô lập (`XQ_NO_SEG_PALACE`): tắt hẳn code palace **vẫn regress** → **không phải code palace, không phải dataset** (user đúng: không thêm bớt gì).
- **Nguyên nhân: inference imgsz.** v5 train@960 → ultralytics mặc định infer board @960; board mask @960 localize góc ngoài **kém hơn @640** (vài bàn lệch 30-60px). **960 là của model items/pieces, KHÔNG phải board seg.**
- **Fix: pin board-seg `imgsz=640`.**

## 5. Kết quả (86-set, mirror-tolerant, `scripts/eval_compare_seg.py`)
| Bước | candidate vs baseline |
|---|---|
| v5 ban đầu (@960) | 48 vs 59 |
| + gate refit #6 | 50 |
| + bỏ refit | 54 |
| + palace-corner RANSAC | 54 |
| + fix `_reduce_to_quad` | 55 |
| **+ pin imgsz 640** | **57-58 vs 58-59 (parity)** |
| + refit chỉ khi cut-off | 57 (bỏ lỗi nặng 13) |

Còn 32,48 lệch (đẹp@960 hỏng@640) = variance độ-phân-giải model nano, không phải bug.

## 6b. KIẾN TRÚC CUỐI CÙNG — 1 MODEL GỘP CHUNG (v5_640)
Sau khi retrain v5 ở **imgsz 640** (train/infer khớp), board mask ngang/hơn model cũ → gộp lại **1 model duy nhất**:
- `board_seg.pt` = `board_seg_v5_640.pt` (2-class) lo CẢ grid + palace orientation.
- Bỏ `palace_seg.pt` + `palace_segmenter`; pipeline lấy palace từ chính `seg.palace_centers`.
- **Kết quả: 59/86 (> baseline cũ 58)**, không regress grid (chỉ 48 lệch; 30 & 83 hồi).
- Deploy + sync `deploy/boarddetection/`. Backup `models/backups/board_seg_v5_640.pt`.

## 6. (lịch sử) Deploy thử v5@960 rồi revert
Phát hiện khi soi visual: deploy v5 làm board model regress GRID nhiều bàn hơn metric FEN-exact báo (vd 23 grid sai, bị che vì đã có 1 piece sai). User chốt: **quay lại grid cũ, palace chỉ xác định chiều**.
- `board_seg.pt` = **board model CŨ** (restore) → GRID.
- `palace_seg.pt` = **v5** → CHỈ lấy palace centroid cho ORIENTATION.
- `build_grid_from_quad` rút gọn: palace chỉ set chiều, KHÔNG refit hình học.
- Pipeline load 2 model; sync `deploy/boarddetection/`.
- **Kết quả: 58/86 (= baseline cũ, KHÔNG regress grid; 23/32/48 hồi) + fix cam-dọc.**
- Backup: `board_seg_predeploy_2026-06-16.pt`, `board_seg_deploy_old_2026-06-16.pt`.
- TODO user đề xuất: thiếu board-conner (cut-off) → palace dựng grid (fallback) — chưa làm.

## 7. Visual review
- `scripts/viz_grid.py` → `test/visual_v5/` (grid + giao điểm + FEN det/gt, prefix OK_/WRONG_).

## 8. Còn lại / chưa làm
- Validate lợi ích thật trên ảnh **cam-dọc / cut-off** (86-set không chứa các ca này).
- Backlog audit chưa làm: #7 luật tướng/sĩ/tượng mạnh hơn (cung-cấm bất kể conf, đếm tướng=1, tướng đối mặt) — lưu ý sĩ/tượng ĐÃ được `_fix_invalid_positions` xoá khi conf<0.7; #8 width FEN; #9 tốt home-cols; #10 điểm-chạm quân; #11 dedupe xe↔mã; #13 tách gate mirror.
- Cân nhắc retrain v5 ở imgsz 640 (khớp inference) để bớt variance 32,48.

## Files mới/sửa
- Mới: `scripts/eval_compare_seg.py`, `scripts/viz_grid.py`, `scripts/diag_board_seg_v5.py`, `docs/ALGORITHM_AUDIT.md`, `docs/SESSION_LOG_2026-06-16.md`, `data/board_seg_v5/`, `models/backups/board_seg_v5.pt`, `models/backups/board_seg_predeploy_2026-06-16.pt`.
- Sửa: `boarddetection/board_segmenter.py`, `boarddetection/item_detector.py`, `boarddetection/pipeline.py`, `boarddetection/models/board_seg.pt` (→v5).
