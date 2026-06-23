# Bench audit — GT corrections + palace-anchored grid (2026-06-23)

Sau khi deploy **items_v19** (synth v3, 136 skin), audit bench 224 ảnh để tìm
đòn bẩy tăng exact. Phát hiện 2 việc: **ground-truth bench hỏng** (lớn nhất) +
**grid perspective** (nhỏ hơn). Model KHÔNG cần train lại cho phần này.

## 1. Ground-truth bench bị SAI — 12 board

Triệu chứng ban đầu: vài board ~30 lỗi, EXTRA=240 "hằng số" giống hệt mọi model.
Soi kỹ: ảnh là bàn ĐẦY quân nhưng `test/bench/ground_truth.txt` ghi thế THƯA →
mọi quân model nhận đúng đều bị tính "dư" (EXTRA ảo) + 0 điểm oan.

Đối chiếu với `test/ground_truth.txt` (nhãn user đã sửa, 86 dòng) → 12 board lệch.
DET model trùng khít test GT ở đa số → xác nhận bench GT sai, test GT đúng.

**12 board đã sửa (= test GT, user duyệt):**
`008, 009, 011, 024, 029, 035, 042, 047, 055, 061, 077, 086`

Ví dụ 061: bench GT ghi 12 quân, ảnh thật 32 quân (khai cuộc gần đủ); grid của
model hoàn hảo (32 quân snap khít, residual 0.15 cell) → lỗi 100% do GT.

> Bài học: bench GT phải audit định kỳ. So nhanh: `grep` diff bench GT vs test GT;
> hoặc cờ `raw_detect >> GT_count` (ảnh đầy mà GT thưa).

## 2. Palace-anchored grid (hypothesis selection)

`build_grid_from_quad` trước đây dựng grid **chỉ từ 4 góc ngoài** board (palace
chỉ dùng xác định chiều). Khi 4 góc ngoài lệch (che/viền gỗ/nghiêng) → grid méo →
quân map sai ô.

Fix (`item_detector.py`): dựng thêm grid ứng viên **fold 8 góc palace + 4 góc board,
findHomography RANSAC**, rồi `_grid_score` chấm theo số quân snap khít → giữ grid
nào quân khớp hơn. **Thiên về baseline 4-góc** (chỉ đổi khi palace-anchored rõ ràng
tốt hơn) → không regress. Đóng góp +1 exact; an toàn cho prod.

Hạn chế: khi baseline quá méo (perspective NẶNG, vd 190), việc gán góc palace vào
toạ độ grid thất bại → không dựng được ứng viên anchored. Xem mục 4.

## 3. Kết quả eval bench (224 ảnh)

| Mốc | v19 exact | MISS | EXTRA | WRONG | tổng lỗi |
|---|---|---|---|---|---|
| Ban đầu (GT hỏng) | 127 | 248 | 237 | 160 | 645 |
| +grid-fix +5 GT | 131 | 213 | 135 | 147 | 495 |
| **+12 GT (hiện tại)** | **136** | **139** | **54** | 130 | **323** |

So model (sau full fix):
| Model | exact | MISS | EXTRA | WRONG |
|---|---|---|---|---|
| v16 (+notok/prod cũ) | 129 | 152 | 58 | 128 |
| v17 synth-v1 | 116 | 180 | 58 | 140 |
| **v19 synth-v3 (DEPLOY)** | **136** | 139 | 54 | 130 |

→ EXTRA sụp 237→54 (≈75% EXTRA cũ là GT ảo). v19 vẫn tốt nhất (+7 vs v16).

## 4. Lỗi còn lại (thật) — hướng tiếp theo

### a) WRONG 130 — **mã ↔ xe là 71/130 (55%)**
`k.xe↔k.mã` 37 + `đ.xe↔đ.mã` 34. Chữ 馬/車 nhầm 2 chiều cả 2 màu. Hướng: synth
**oversample riêng mã+xe** nhiều font; hoặc kiểm tra imgsz inference. Đòn bẩy lớn
nhất cho WRONG.

### b) MISS 139 — rải đều, top đ.pháo 24 / k.mã 19. Cần đa dạng data thật hơn.

### c) Grid perspective nặng — 190, 116
Bàn chụp xiên ngoài trời: board_seg quad méo, baseline quá lệch → palace assign
fail (190: 32 quân detect nhưng chỉ 23 map). Cần bootstrap palace KHÔNG phụ thuộc
baseline (vd dùng 2 palace-center + chiều để gán trước, rồi RANSAC 8 góc palace).
Số board kiểu này ít (~2-4).

## 5. Scripts
- `scripts/grid_debug.py <ids>` → overlay seg quad (đỏ) + palace (vàng) + grid
  project (xanh) vào `test/bench/_grid_debug/` để soi mắt.
- `scripts/piece_errors.py` → per-class MISS/EXTRA/WRONG + `test/bench/piece_errors.txt`.
- `scripts/eval_bench.py` → bảng so model (MODELS dict).

## 6. Trạng thái
- **prod = items_v19** (deploy 2026-06-23), bench thật **136/224**.
- grid-fix đã trong pipeline (prod hưởng lợi).
- bench GT đã sạch 12 board; có thể còn lỗi GT nhẹ ngoài 86 dòng đầu (chưa có test GT đối chiếu).
