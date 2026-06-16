# Audit giải thuật — Nhận diện cờ tướng (2026-06-16)

Audit đa-agent đối kháng (7 chiều, 53 agent, **33 finding đã verify**).
Run nguồn: `wf_abfa59b1-b8b`. Mỗi finding bên dưới đều được kiểm chứng lại trên code thật.

## Tóm tắt

Vấn đề tác động lớn nhất: seg-grid là **đường chính** (`pipeline.py:174`) nhưng **không bao giờ
được sanity-check**, trong khi `_grid_from_4_corners` (`item_detector.py:1006-1043`) nhận tham số
`image_shape` nhưng **bỏ qua hoàn toàn** — không có guard về biên/suy biến. Kết hợp với việc mask bị
cắt mép (ảnh chụp thiếu bàn), một quad **sai-nhưng-không-None** sẽ **chặn luôn fallback landmark**
(`pipeline.py:183`). Một lỗ hổng đó gây ra các thất bại tệ nhất trên ảnh thật (cut-off / phối cảnh),
nên gate validate + xử lý cut-off là ưu tiên số 1, và **sửa được mà không cần v5**.

Bug xoay 90° (`board_segmenter.py:69-72`) là thật và đã xác nhận; bản sửa sạch (dùng trục palace)
cần wire `board_seg_v5`. Bản sửa tạm: ngừng relabel ở segmenter, để `_corners_to_correspondences`
(vốn đã ưu tiên trục palace, `item_detector.py:855-887`) làm chủ quyết định trục hàng, tránh 2 tầng
đánh nhau.

Đính chính: `detect_board_orientation` (`fen_generator.py:385-444`) **không** phải flip ngây thơ theo
1 tướng — nó đã vote đa số 3 tín hiệu. Rủi ro còn lại chỉ ở ngưỡng nửa-bàn khi tướng đơn độc là tín
hiệu duy nhất.

## Danh sách fix ưu tiên

### #1 [cao] Validate seg-grid + dùng `image_shape`; loại lattice tràn-biên để fallback landmark chạy được
- **Cần v5:** Không
- **Vị trí:** `item_detector.py:1006-1043` (image_shape không dùng) + `pipeline.py:168-189`
- **Sửa:** Trong `_grid_from_4_corners`, sau khi chiếu grid: trả None nếu `cell_w<5`/`cell_h<5`, nếu `|t[2]|<1e-6` khi chiếu, hoặc nếu quá nhiều trong 90 điểm rơi ngoài rect mở rộng `[-0.25w,1.25w]x[-0.25h,1.25h]`. Trong `pipeline.recognize`, ép `grid=None` (bắt fallback `build_grid_from_landmarks`) khi bất kỳ góc quad nằm trong ~3px mép ảnh, hoặc tỉ lệ cạnh cols:rows bất hợp lý với bàn ~9:10; append lỗi rõ ràng "board cut off / quad bất hợp lý".

### #2 [trung] Xây confidence/alternate từ assignment thật của `map_pieces_to_grid`, không re-derive `get_nearest_cell`
- **Cần v5:** Không
- **Vị trí:** `pipeline.py:213-247` (to_cell) vs `fen_generator.py:146-154` + `piece_positions`
- **Sửa:** `map_pieces_to_grid` đã ghi (row,col) cuối của từng quân trong `piece_positions` (kể cả sau khi `_nearest_free_cell` dời chỗ vì va chạm). Truyền map đó ra ngoài và dựng `piece_confidences`/alternates từ nó (áp transform flip lên cell đã biết), thay vì `to_cell` tự tính lại. Khi va chạm thì giữ max confidence để 2 quân snap về 1 ô không ghi đè ngầm.

### #3 [cao] Ngừng segmenter relabel 90° theo tỉ-lệ-cạnh; giao quyết định trục hàng cho tầng correspondence có-palace
- **Cần v5:** Có (bản final)
- **Vị trí:** `board_segmenter.py:69-72` (tạm) → wire `board_seg_v5` (final)
- **Sửa:** TẠM (không v5): bỏ swap `cols_edge>1.05*rows_edge` ở `board_segmenter.py:71-72`, phát góc theo thứ tự vòng-góc, để `_corners_to_correspondences` (đã ưu tiên vector tâm 2 cửu cung / PCA palace làm trục rank) là chủ duy nhất của trục hàng. FINAL (cần v5): wire `MODELS_DIR/board_seg_v5.pt` ở `pipeline.py:90`, truyền tâm 2 mask palace vào `get_board_quad`/`build_grid_from_quad`, đặt `rows_axis = vector palace→palace`; chỉ dùng heuristic cạnh khi phát hiện <2 palace.

### #4 [trung] Thay sắp xếp góc theo x±y bằng sắp theo góc-quay (tilt-robust)
- **Cần v5:** Không
- **Vị trí:** `board_segmenter.py:54-59` + `item_detector.py:910-912`
- **Sửa:** Trong `get_board_quad` sắp 4 góc theo `atan2` quanh tâm thành vòng CW, rồi xoay sao cho góc tổng-nhỏ-nhất đứng đầu (robust ở tilt ~45° nơi x±y bị tie và hàm hiện trả None ở 58-59). Trong `_corners_to_correspondences` (910), khi va chạm nhãn thì KHÔNG trả `standard_portrait()`; giải xác định bằng sort theo `row_proj`, hoặc trả None để caller fallback sang `build_grid_from_landmarks` thay vì bịa grid thẳng đứng.

### #5 [cao] Thêm đường dựng grid neo-bằng-cửu-cung cho bàn cut-off (không chỉ refine grid đã dựng)
- **Cần v5:** Có
- **Vị trí:** `item_detector.py:914-970` (build_grid_from_quad) + `_grid_from_correspondences:972-1004`
- **Sửa:** Khi guard #1 cờ ≥1 góc bị cắt, dựng grid CHÍNH từ neo cửu cung qua `_grid_from_correspondences` dùng chỉ các góc chưa-cắt + palace-centers (4,1)/(4,8) + palace-corners (3,2)(5,2)(3,7)(5,7) + palace-bottoms (3,0)(5,0)(3,9)(5,9) (tới 10 neo nội, trải cột 3-5 hàng 0-9). Đổi `_grid_from_correspondences` từ `findHomography(...,0)` sang `RANSAC/LMEDS` để loại 1 palace gán sai như outlier; chỉ thêm neo palace-center khi khoảng cách tới target gán rõ ràng nhỏ hơn target kia.

### #6 [trung] Gate refit palace-center để 1 neo gán sai không thay được grid 4-góc tốt
- **Cần v5:** Không
- **Vị trí:** `item_detector.py:943-970` (refit trong build_grid_from_quad)
- **Sửa:** Chỉ trả `refined` nếu nó giữ cả 4 góc trong vài px của dst VÀ variance kích thước ô ≤ grid gốc; ngược lại giữ `grid`. Gán nearest-target hiện tại (948-963) có thể map palace-center giữa-bàn sang nhầm (4,1)/(4,8) trên grid méo phối cảnh, rồi least-squares warp cả lattice theo neo sai.

### #7 [trung] Thêm pass hợp-lệ: đếm tướng / cung-cấm / tướng-đối-mặt khi inference
- **Cần v5:** Không
- **Vị trí:** `rules_validator.py:55-84` (validate_and_correct) + `pipeline.py:245-250`
- **Sửa:** Thêm pass sau sửa: (a) nếu số K/k ≠ 1 → cờ lỗi, có thể promote ứng viên tướng confidence cao nhất từ alternates; (b) ép cung-cấm cho K/k bất kể confidence; (c) check tướng đối mặt (cùng cột, không quân chắn). Tối thiểu chạy `validate_fen` trên FEN cuối và append lỗi vào `RecognitionResult.errors`.

### #8 [trung] Ép đủ 9 cột trong `parse_fen`/`validate_fen`
- **Cần v5:** Không
- **Vị trí:** `fen_generator.py:281-292` (parse_fen) + 294-333 (validate_fen)
- **Sửa:** `parse_fen` đang lặng lẽ bỏ quân quá cột 8 (guard `if col_idx < GRID_COLS` ở 287) và để hàng thiếu cột trống dở. Sau mỗi hàng assert tổng cột == GRID_COLS (raise ValueError nếu sai), thêm check width mỗi hàng trong `validate_fen` trước khi đếm quân.

### #9 [thấp] Hiện thực ràng buộc tốt chết theo `home_cols`
- **Cần v5:** Không
- **Vị trí:** `rules_validator.py:94-100` (_is_legal_cell) + 185-194
- **Sửa:** `_is_legal_cell` cho tốt chỉ check `min_row<=row<=max_row`; config `home_cols={0,2,4,6,8}`/`home_rows` (47-48) không hề dùng. Khi tốt chưa qua sông & ở `home_rows` thì ép `col in home_cols`; chỉ cho cột tùy ý sau khi qua sông. Áp cùng logic (gate theo low-conf) ở `_fix_invalid_positions`.

### #10 [trung] Dùng điểm-chạm đáy-bbox, không phải tâm hình học, khi snap quân→ô trên ảnh nghiêng
- **Cần v5:** Không
- **Vị trí:** `item_detector.py:118` (cx,cy tâm bbox) → get_nearest_cell → `fen_generator.py:129-130`
- **Sửa:** bbox quân (đĩa cao) đặt tâm hình học cao hơn điểm chạm bàn; ảnh xiên làm lệch về phía cam, snap nhầm giao điểm. Dùng `contact_y = y1 + (y2-y1)*k` với k~0.7-0.8 (hiệu chỉnh trên test set), hoặc snap đáy-bbox qua grid phối cảnh. Kèm 2-pass `snap_ratio` (0.6 rồi tới ~0.9 cho quân conf-cao chưa gán ở rìa) và log quân conf-cao bị drop.

### #11 [trung] Dedupe quân theo khoảng-cách-tâm trước khi map grid; siết `_nearest_free_cell`
- **Cần v5:** Không
- **Vị trí:** `piece_detector.py:249-293` (NMS chỉ IoU) → `fen_generator.py:146-169`
- **Sửa:** NMS IoU không dedupe 2 box khác-class (lỗi xe↔mã hay gặp) cùng tâm 1 giao điểm. Thêm pass khoảng-cách-tâm (như `_dedupe_landmarks` ở `item_detector.py:1207`): nếu 2 detection cách <~0.5 ô, giữ conf cao hơn, hạ cái kia thành alternate. Trong `_nearest_free_cell` chỉ dời khi khoảng cách gốc <~0.35 ô VÀ ô trống liền kề, ngược lại drop.

### #12 [thấp] Mạnh hóa tín hiệu orientation 1-tướng + siết `_snap_quad_to_corners`
- **Cần v5:** Không
- **Vị trí:** `fen_generator.py:419-427` + `item_detector.py:889-899` (SIGN) + `pipeline.py:274-300`
- **Sửa:** Khi tướng đơn là tín hiệu duy nhất, ngưỡng row>=5/<5 mong manh → yêu cầu tướng ở rõ nửa sai (đỏ<=2 / đen>=7) mới vote flip. Trong SIGN cho dùng 1 tướng (K dương / k âm) trước khi cần 2+ quân mỗi màu; cờ low-conf khi không có tín hiệu. Trong `_snap_quad_to_corners` cho matching tối ưu toàn cục (sort theo khoảng cách mọi cặp, không greedy break ở 299), scale ngưỡng 50px → ~0.4*ô, loại 1 góc snap lệch với phần còn lại.

### #13 [trung] Cờ fallback interpolation là low-conf + tách gate deploy mirror-tolerant
- **Cần v5:** Không
- **Vị trí:** `pipeline.py:195-199` + `scripts/evaluate_fendata.py:49-67`
- **Sửa:** Khi grid là None và chạy `map_pieces_to_grid_by_interpolation`, ép `confidence~0` (hoặc cờ failed) để FEN-gate loại kết quả đoán. Eval/deploy gate hiện tính FEN lật-ngang là đúng (pred_m) trong khi runtime KHÔNG lật ngang (`pipeline.py:201-207`) → báo exact-match và mirror-match riêng, gate deploy chỉ theo exact-match.

## Khoảng trống cần test runtime trên ảnh thật

- `board_seg_v5.pt` CHƯA wire (`pipeline.py:90` vẫn load `board_seg.pt`) → các fix palace (#3, #5) chưa validate được cho tới khi wire v5 và xác nhận v5 phát đúng 2 palace/bàn.
- Chưa có test ảnh cut-off (1-2 góc thật ngoài khung) để xác nhận guard biên + dựng-từ-palace (#1, #5) thực sự kích hoạt fallback và ra grid đúng.
- Chưa có ảnh phối cảnh dọc (trục 9-cột chiếu dài hơn 10-hàng) để chứng minh bỏ relabel segmenter (#3) không regress bàn ngang upright.
- Chưa có ảnh tilt ~45° kim cương để test tie sắp-góc (#4).
- Chưa có ảnh tàn cuộc thưa (tướng đơn + xe) để test orientation SIGN / vote 1-tướng (#12).
- Hệ số điểm-chạm k (#10) chưa hiệu chỉnh; cần sweep trên test set.
- Lỗi xe↔mã cùng giao điểm (#11) cần ảnh thật tái hiện 2 box khác-class IoU<0.35.
- Toàn pipeline đang chấm mirror-tolerant (#13) → exact-match thật trên 86 ảnh chưa biết tới khi tách metric.
