# Synthetic Assets Report — quân cờ & bàn cờ (để check)

_Cập nhật 2026-06-22. Dùng cho synthetic data generation (synth_gen.py)._

## 1. QUÂN CỜ (piece crops)

### Library chuẩn: `data/piece_crops_all/<class_name>/*.png`
- **10,276 crop** RGBA PNG, ASCII path (đã chuẩn hóa — fix bug cv2 đọc path tiếng Trung)
- **734 crop/class × 14 class** (cân bằng tuyệt đối)
- Mỗi crop/class = **134 digital + 600 real**

| Nguồn | Số skin | Chi tiết |
|---|---|---|
| Digital PNG_hires | 18 skin | 393px, tên file `{b\|r}{k,a,b,n,r,c,p}.png` |
| Digital SVG (raster→PNG 400px) | 54 skin | resvg, unicode-safe |
| Digital _lowres_54px | 62 skin (load ok/69) | 54px, font khác (3D楷, 上海杯...) |
| **Tổng digital** | **134 skin** | tag `d000`–`d133`, map tên gốc (Trung) ở `skin_map.json` |
| Real harvest (từ items_v16 bbox) | 600/class = 8400 | elliptical alpha, texture gỗ/nhựa thật |

> Nguồn skin: `download/BO_COTUONG_CLEAN/{PNG_hires(18), SVG_vector(54), _lowres_54px(69)}`
> = 141 set distinct (không trùng). 134 load sạch. `_PREVIEW` chỉ là html/ảnh preview, bỏ.

→ **Check:** mở `data/piece_crops_all/<class>/` xem crop; `skin_map.json` để biết d### là skin nào.

## 2. BÀN CỜ (empty boards)

### Dùng để sinh ảnh: `data/empty_boards/` — **51 bàn grid ĐÚNG**
- Đánh số (`manifest.csv`: num,type,original). corners.json = 4 góc đã QC mỗi bàn.
- Real nghiêng (gỗ thật, nền thật) + digital phẳng (có viền).

### Đã LOẠI (grid sai/lệch/board_seg fail) — KHÔNG dùng sinh ảnh:
`002, 006, 009, 011, 013, 014, 016, 020, 021, 023, 025, 026, 027, 030, 035, 052, 053`
(lệch/sai) + `024, 055, 057, 058, 063, 064, 066, 068, 074` (board_seg không thấy).
→ Tổng QC 77 → giữ **51**.

### Pool train board-seg sau: `data/board_seg_pool/images/` (76 ảnh, giữ cả bàn khó)
→ **Check:** overlay grid ở `data/empty_boards/_grid_qc/<num>.jpg`.

## 3. FEN (thế cờ) — ⚠️ CẦN XỬ LÝ

| Nguồn | Unique parse-ok | Dùng được? |
|---|---|---|
| `data/fendata/fendata/labels.csv` | **chỉ 17** (786 dòng trùng vị trí) | quá ít |
| test/bench/ground_truth.txt (224) | ~210 | ❌ **KHÔNG** — là tập EVAL → leakage |
| test/ground_truth.txt (85) | ~85 | ❌ nếu dùng eval |

**Vấn đề:** chưa đủ FEN train đa dạng (chỉ 17 an toàn). Dùng bench/test = gian lận điểm.
**Hướng fix:** **derive FEN từ label items_v16** (4197 ảnh = vị trí training, không phải eval)
→ convert YOLO label → FEN → hàng nghìn thế cờ thật. (chưa làm)

## 4. BUGS đã phát hiện & fix

| # | Bug | Hậu quả | Fix |
|---|---|---|---|
| 1 | cv2.imread/imwrite **fail path non-ASCII** (Windows) | 18 PNG skin tên Trung load **0**; 47 SVG skin imwrite fail thầm → synth v1 chỉ ~7 skin | `imread_u/imwrite_u` (np.fromfile+imdecode / imencode+tofile). Re-raster SVG OK. Consolidate ra ASCII library |
| 2 | FEN csv chỉ 17 unique | synth v1 chỉ 17 thế cờ → ít đa dạng vị trí | dùng FEN derive từ items_v16 (chưa làm) |
| 3 | DataLoader worker crash (shared-mem Windows) khi train | train v17 chết epoch 1 | `--workers 2 --batch-size 12` (đã thêm `--workers` vào train_items.py) |
| 4 | SVG `width=mm` → resvg "invalid size" | raster fail | replace `mm`→`px` trước raster |

## 5. Trạng thái synth v1 (đang train v17)
- v17 đang train trên synth v1 — **synth v1 BỊ DÍNH bug 1+2** (chỉ ~7 skin, 17 FEN).
- → Kết quả v17 chỉ là data-point tham khảo, KHÔNG phải đánh giá đúng tiềm năng synthetic.
- **Tối nay regen v2** với library chuẩn (72 skin + 8400 real) + FEN nhiều hơn → train lại mới fair.

## 6. Lệnh regen v2 (sau khi có thêm FEN)
```bash
python scripts/consolidate_crops.py        # (đã chạy → piece_crops_all 9408)
python scripts/synth_gen.py --crops-root data/piece_crops_all \
  --boards data/empty_boards --corners data/empty_boards/corners.json \
  --fens <FEN_SOURCE_ITEMS_V16> --out data/items_synth_v2 --n 1500 --layout split
# rồi merge items_v16 → split → train --workers 2 --batch-size 12 → eval_bench
```
