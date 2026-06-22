# Synthetic Data Generation — Xiangqi (ghép quân vào bàn)

> Mục tiêu: sinh ảnh train bằng cách ghép quân (digital skins + crop thật) vào ảnh
> bàn TRỐNG theo FEN thật, qua homography của bàn → **label 18-class hoàn hảo tự động**.
> Trộn với data thật (items_v16) → tăng đa dạng font/màu + cân bằng class → kỳ vọng
> vượt bench v16 (118/224 exact). Prove nhỏ trước, gate bằng bench.

## Vì sao
Bench 224 sai chủ yếu: MISS 262, EXTRA 240, WRONG 164. WRONG (nhầm loại/màu) + imbalance
class là thứ data đa dạng giải quyết tốt nhất. Synthetic cho label miễn phí + đa dạng vô hạn.

## Nguyên lý hình học (đã verify)
```
src = float32([[0,0],[8,0],[0,9],[8,9]])   # grid (col,row): tl,tr,bl,br
dst = float32([tl,tr,bl,br])                # 4 góc bàn (pixel) từ corners.json
H   = cv2.getPerspectiveTransform(src,dst)  # H @ [col,row,1] → pixel
```
FEN row r → grid row r ⇒ không cần đoán orientation (ta tự kiểm soát khi đặt quân).
GRID_COLS=9 (0-8), GRID_ROWS=10 (0-9).

## 18-class scheme (settings.ITEM_CLASSES)
0-6 black {advisor,cannon,chariot,elephant,general,horse,soldier} ·
7 board-conner · 8 palace-bottom · 9 palace-center · 10 palace-conner ·
11-17 red {advisor,cannon,chariot,elephant,general,horse,soldier}
Landmark cells: conner [(0,0),(8,0),(0,9),(8,9)] · palace-bottom [(3,0),(5,0),(3,9),(5,9)] ·
palace-conner [(3,2),(5,2),(3,7),(5,7)] · palace-center [(4,1),(4,8)].

## Assets
| Loại | Nguồn | Số lượng |
|---|---|---|
| Empty board (grid ĐÚNG, đã QC) | `data/empty_boards/` (real angled + digital) | **53** (đánh số, `manifest.csv`) |
| Empty board pool (train board-seg sau) | `data/board_seg_pool/` | 76 (giữ cả bàn khó) |
| Quân digital PNG | `download/BO_COTUONG_CLEAN/PNG_hires/<skin>/{b\|r}{k,a,b,n,r,c,p}.png` | 252 (18 skin) |
| Quân digital SVG | `download/BO_COTUONG_CLEAN/SVG_vector/` | 756 (54 skin) — **PENDING raster** |
| Quân THẬT harvest | `data/piece_crops_real/<class>/` | **8400** (600/class, cân bằng) |
| FEN thật | `data/fendata/fendata/labels.csv` (+ bench/test gt) | ~786 (+224+85) |

## Scripts (đã build)
| Script | Việc |
|---|---|
| `scripts/viz_board_grid.py` | Grid QC: board_seg → overlay lưới 9×10 lên mỗi bàn → `_grid_qc/` + `corners.json` |
| `scripts/rm_board.py <num...>` | Xóa bàn grid sai khỏi empty_boards (+qc+corners). Giữ board_seg_pool |
| `scripts/harvest_crops.py` | Cắt crop quân thật từ items_v16 bbox → RGBA (elliptical feathered alpha), 600/class |
| `scripts/synth_gen.py` | Generator chính: ghép quân + sinh label 18-class |

## Pipeline synth_gen
1. Load crop pool: digital PNG skins + real harvest (RGBA), index theo class id
2. Mỗi ảnh: chọn bàn (corners.json) → H; chọn FEN thật
3. Mỗi ô có quân: `cell_quad(col,row,half=0.45)` project qua H → warpPerspective crop → alpha
   composite. **Draw order**: sort theo projected-y (quân gần camera đè lên)
4. Realism: rotate ±4°, scale 0.92-1.05, brightness/hue match nền, feather, drop-shadow, JPEG q78-93
5. Label: AABB của chính cell_quad (label↔pixel nhất quán) + 4 landmark vị trí cố định
6. Output `--layout split` (train/images+labels) + data.yaml 18-class

## Quy trình chạy (gate bằng bench)
```bash
# 0. GRID QC (đã làm) — review _grid_qc/, xóa bàn sai bằng rm_board.py
python scripts/viz_board_grid.py

# 1. harvest crop thật (đã làm → 8400 crop)
python scripts/harvest_crops.py --src data/items_v16 --out data/piece_crops_real --per-class 600

# 2. sinh ~1000 ảnh synthetic
python scripts/synth_gen.py --boards data/empty_boards --corners data/empty_boards/corners.json \
  --png-root download/BO_COTUONG_CLEAN/PNG_hires --real-root data/piece_crops_real \
  --fens data/fendata/fendata/labels.csv --out data/items_synth_v1 --n 1000 --seed 42 --layout split

# 3. merge với base thật items_v16
python scripts/merge_datasets.py --out data/items_v17_synth --src data/items_v16 --src data/items_synth_v1

# 4. split + data.yaml
python scripts/split_items.py --dir data/items_v17_synth

# 5. train (KHÔNG auto-deploy)
python scripts/train_items.py --data data/items_v17_synth/data.yaml --name items_v17_synth --no-deploy

# 6. eval vs v16 (118 baseline) — thêm backup vào MODELS trong eval_bench.py
python scripts/eval_bench.py
```
**Gate:** bench exact ≥118 và MISS/EXTRA/WRONG không tệ đi → scale Phase 2 (~2500). Deploy thủ
công (copy backup → boarddetection/models/items.pt) chỉ khi vượt. Tỉ lệ synthetic ≤50% pool.

## Trạng thái hiện tại (2026-06-22)
- ✅ Grid QC xong: 77 bàn → giữ **53 bàn grid đúng** (xóa bàn lệch/sai + 9 bàn board_seg fail)
- ✅ board_seg_pool (76 ảnh) để train board-seg sau
- ✅ harvest_crops: 8400 crop thật cân bằng
- ✅ synth_gen.py build xong (chạy với 18 PNG skin + 8400 real)
- ⏳ PENDING: rasterize 756 SVG quân (54 skin) — resvg_py báo "invalid size" (SVG thiếu width/height),
  cần inject size từ viewBox. Chưa critical (18 PNG skin + 8400 real đã đa dạng)
- ⏭️ NEXT: chạy synth_gen ~1000 → merge → train v17 → eval bench

## Rủi ro + giảm thiểu
1. Sim→real gap (digital phẳng vs gỗ thật): trộn real crop (8400), giữ synthetic ≤50%,
   brightness/hue match. Watch: val mAP lên nhưng bench xuống = overfit → giảm tỉ lệ
2. Bàn nghiêng: chỉ dùng 53 bàn grid ĐÚNG (đã loại bàn lệch); shadow + brightness match
3. Label lệch pixel: 1 nguồn `cell_quad` cho cả paste lẫn label
4. half=0.45 (footprint 90% cell): verify với label thật, tune nếu lệch
5. FEN parse: hardened (digit-run gộp, leading-zero `02`), skip FEN ≠ 90 ô
```
```
