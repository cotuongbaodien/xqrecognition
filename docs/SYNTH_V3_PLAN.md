# Synth v3 — Phân tích root-cause & kế hoạch làm lại dataset (CHỜ APPROVE)

> Tài liệu để review trước khi thực thi. Không code/regen cho tới khi được duyệt.

## Vì sao v17 (synth v1) TỆ HƠN — eval bench 224 ảnh
| Model | exact | MISS | EXTRA | WRONG |
|---|---|---|---|---|
| v14 (old) | 120 | 262 | 240 | 164 |
| v16 (+notok, **prod**) | 118 | 259 | 240 | 161 |
| v17 (synth v1) | **109** ↓ | **288** ↑ | 240 | 169 |

Synth v1 làm **MISS tăng** (sót quân thật). → Đúng nhận định: **sai CÁCH TẠO**, không sai ý tưởng.
Cut-paste synthetic là kỹ thuật chuẩn; có 141 bộ quân + 60 bàn + 8400 crop thật thì phải giúp được.

## ROOT CAUSES (có bằng chứng đo)

1. **[ĐO ĐƯỢC] Quân synth DẸT** — label real h/w=1.14 (cao, vì quân 3D nghiêng), synth h/w=0.93,
   **h synth = 0.81× thật**. Do `cell_quad(half=0.45)` footprint vuông → mất chiều cao 3D → model
   học quân thấp → MISS quân thật. **Fix: footprint cao (piece_height ~0.2 cell).**

2. **[XÁC NHẬN] v17 chỉ ~7 skin + 17 FEN** — cv2 Windows fail path tiếng Trung (18 PNG load 0,
   SVG 7 skin) + fendata 17 thế trùng → 1000 ảnh gần như lặp → overfit. **Đã fix:** unicode-safe IO +
   consolidate 134 skin + gen 4000 FEN.

3. **Blend 1 kiểu** (feather+shadow cố định) → model bám "vết ghép". **Fix: random multi-blend**
   (hard / feather σ ngẫu nhiên / Poisson seamlessClone).

4. **Digital phẳng dán bàn thật nghiêng = mismatch**. **Fix: match crop↔board** (bàn thật→crop thật,
   bàn digital→crop digital).

5. Real crop bị footprint ép dẹt → fix #1 giải quyết.

6. FEN random ≠ thật (nhỏ với detection) — optional.

7. **[CLUE] EXTRA=240 giống hệt cả 3 model** → lỗi CHUNG = board_seg/grid, không phải items model.
   Fix grid có thể cắt 240 EXTRA → tăng exact bất kể synth. **Hướng riêng giá trị cao.**

## Phân tích từng SET

### Quân (141 skin + real)
| Set | Số | Res | Dùng cho | Weight |
|---|---|---|---|---|
| Real harvest | 8400 (600/cls) | thật | bàn THẬT (khớp domain bench) | cao trên bàn real |
| PNG_hires | 18 skin | 393px | bàn digital + đa dạng font | TB |
| SVG raster | 54 skin | 400px | bàn digital + font | TB |
| lowres_54px | 69 skin | 54px (mờ) | thêm font, chỉ bàn digital | thấp ≤10% |
> 141 distinct (18+54+69). Consolidate được 134 → **khôi phục 7 lowres còn thiếu**.

### Bàn (60: 45 real + 15 digital) — geometry đo từ corners.json
| Loại | Số | Perspective | Đa dạng góc |
|---|---|---|---|
| Real nghiêng | 45 | 1.00–1.63 | thẳng 20 / vừa 14 / mạnh 11 |
| Digital phẳng | 15 | ~1.00 | top-down đều |
> Bàn đa dạng góc TỐT, khớp bench. Vấn đề ở cách ghép, không thiếu bàn.

## KẾ HOẠCH v3 (sửa scripts/synth_gen.py + consolidate)
1. Footprint cao `piece_height≈0.2` → aspect khớp real (~1.14), hết lệch 0.81×.
2. Multi-blend ngẫu nhiên mỗi quân.
3. Crop↔board matching: bàn real→real 0.8/digital 0.2; bàn digital→digital 0.9/lowres 0.1.
4. Board sampling 70% real / 30% digital.
5. Khôi phục 7 lowres skin → đủ 141.
6. Tách pool theo res (cap lowres).

## VALIDATION (gate)
1. Đo lại scale: synth h/w ≈ real (~1.14).
2. Visual A/B montage: quân synth vs quân thật cùng class/bàn.
3. (optional) classifier real-vs-synth: nếu phân biệt quá dễ → artifact còn nặng.
4. **eval_bench v3 vs v16(118)** + per-class MISS/WRONG — thước đo QUYẾT ĐỊNH (không dùng val mAP).
   Gate: exact ≥118, MISS không tăng. Tỉ lệ synth bắt đầu ~30%.

## THỰC THI sau approve
1. (đã làm) cancel gen 4000.
2. khôi phục 7 lowres + re-consolidate → 141 skin.
3. sửa synth_gen (height/blend/matching/pool).
4. regen v3 4000 (70/30) → đo scale.
5. merge items_v16 → split → train (workers 2, batch 12, imgsz 960, --no-deploy).
6. eval_bench → so v16. Pass thì cân nhắc deploy/scale.

## Trạng thái hiện tại
- prod = v16+notok (118). v17 (109) KHÔNG deploy.
- Assets sẵn: 134/141 skin consolidated, 8400 real crop, 60 bàn QC, 4000 FEN.
- CHỜ APPROVE để chạy bước 2-6.

---
## ✅ READY (chuẩn bị xong — tối chạy train)
- `data/items_v19`: train 6557 / valid 1229 / test 411 (4197 real v16 + 4000 synth v3), 18-class
- synth v3 fixes đã áp: scale h/w 1.08 (real 1.14), multi-blend, crop-board matching, 136 skin, 4000 FEN
- GPU free. prod = v16+notok (118), v19 train --no-deploy (gate bench trước khi deploy)

### Lệnh TRAIN (chạy tối nay):
```
python scripts/train_items.py --data data/items_v19/data.yaml --name items_v19 \
  --img-size 960 --batch-size 12 --workers 2 --no-deploy
```
~5h (hoặc early-stop). Sau train → eval:
```
# thêm "v19": "models/backups/items_v19.pt" vào MODELS trong scripts/eval_bench.py rồi:
python scripts/eval_bench.py
```
Gate: exact ≥118 (v16) và MISS không tăng → cân nhắc deploy/scale. Nếu vẫn ≤118 → xem hướng EXTRA=240 (grid).
