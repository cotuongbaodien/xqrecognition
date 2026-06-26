# Weekly active-learning ingest — quy trình + checklist

Mỗi tuần lấy ảnh user submit từ server → model hiện tại (items_v20) đọc tạo
pseudo-label → **1 review chung** để sửa nhãn → gộp vào trainset → retrain.
Mục tiêu 3 tháng, mỗi tuần 1 vòng, tiến tới ~10–15k ảnh.

```
download/ocrservice/2026-WNN/images  ──► items.pt detect (conf≥0.25, imgsz960)
        │                                       │ pseudo-label (YOLO)
        ▼                                       ▼
   (copy từ server)              items_v20/incoming/{images,labels}/  ← STAGING (chưa train)
                                                │ build gallery
                                                ▼
                          data/label_review_incoming/<N.tag>/sheet_*.jpg
                                                │ người review + sửa
                                  ┌─────────────┴─────────────┐
                                  ▼                           ▼
                          merge → train/ (retrain)     pick → test/bench (giữ riêng)
```

---

## ✅ CHECKLIST mỗi tuần

- [ ] **1. Copy ảnh tuần mới** từ server vào `download/ocrservice/2026-WNN/images/`
- [ ] **2. Ingest** (GPU rảnh — detect imgsz960 nặng):
      `python scripts/weekly_ingest.py --input download/ocrservice/2026-WNN`
      (hoặc trỏ `download/ocrservice` để nuốt nhiều tuần 1 lần → 1 review chung)
- [ ] **3. Review từng quân** — mở sheet trong `data/label_review_incoming/` (bảng dưới)
- [ ] **4. Báo sửa** → tôi chạy `apply_review.py <tag> '<idx>=<class>' --dir data/label_review_incoming`
- [ ] **5. (Tùy chọn) Chọn ảnh làm TEST** trước khi merge (xem mục "Mở rộng test")
- [ ] **6. Merge** vào train: `python scripts/weekly_ingest.py --merge`
- [ ] **7. Retrain** → A/B trên bench 243 → deploy nếu tốt hơn

---

## 🔎 Review TỪNG QUÂN ở đâu — `data/label_review_incoming/<folder>/sheet_NNN.jpg`

| Quân | Chữ | Folder | Tag (gõ khi báo) | Hay BỊ LẪN với |
|---|---|---|---|---|
| Xe đen   | 車 | `1.xeden`   | `xeden`   | mã (馬) |
| Mã đen   | 馬 | `2.maden`   | `maden`   | xe (車) |
| Xe đỏ    | 俥 | `3.xedo`    | `xedo`    | mã |
| Mã đỏ    | 傌 | `4.mado`    | `mado`    | xe |
| Pháo đen | 包 | `5.phaoden` | `phaoden` | tốt, sĩ |
| Pháo đỏ  | 炮 | `6.phaodo`  | `phaodo`  | tốt |
| **Tượng** đen | 象 | `7.tuongden` | `tuongden` | **vua (将)** |
| **Tượng** đỏ  | 相 | `8.tuongdo`  | `tuongdo`  | **vua (帥)** |
| Tốt đen  | 卒 | `9.totden`  | `totden`  | pháo, sĩ |
| Tốt đỏ   | 兵 | `10.totdo`  | `totdo`   | pháo |
| Sĩ đen   | 士 | `11.siden`  | `siden`   | tượng, tốt |
| Sĩ đỏ    | 仕 | `12.sido`   | `sido`    | tượng |
| **Vua** đen | 将 | `13.soaiden` | `soaiden` (gõ `vua` cũng được) | **tượng (象)** |
| **Vua** đỏ  | 帥 | `14.soaido`  | `soaido`  (gõ `vua` cũng được) | **tượng (相)** |

> Mỗi sheet 100 ô (10×10), góc trên-trái mỗi ô có **số idx (màu hồng)**. Trong
> 1 folder, tất cả ô PHẢI là đúng quân đó — ô nào sai → ghi lại `idx`.

### Cách báo sửa (tôi áp bằng `apply_review.py`)
- 1 ô sai: `tuongden 7=vua` (ô 7 trong gallery tượng-đen thực ra là vua)
- nhiều ô: `xeden '5,12,30=ma 41=phao'`
- xóa ô rác (không phải quân / cắt lỗi): `totden '3,9=bo'`  (bo = đánh dấu xóa)
- token: `xe ma phao si tuong vua tot` + màu mặc định theo folder (override `xe-do`, `phao-den`)

**Mẹo ưu tiên:** chú trọng 2 cặp hay sai — `tuongden/tuongdo` ↔ `soaiden/soaido`
(tượng↔vua) và `xe ↔ ma`. Các quân khác (tốt/pháo) ít lẫn, lướt nhanh.

---

## 🧪 Mở rộng TEST set (chọn ảnh làm bench)

Bench `test/bench` (hiện 243 ảnh có GT-FEN) là thước đo deploy. Mỗi tuần nên
trích **~15–25 ảnh** ra làm test để bench lớn + đa dạng dần.

**Tiêu chí chọn:** ưu tiên (a) **style/bàn MỚI** chưa có trong bench, (b) vài ca
**KHÓ** (model dễ sai), (c) bàn đầy quân (nhiều thứ để chấm). Tránh trùng/lặp.

**Quy trình:**
1. `python scripts/pick_test_candidates.py --n 20`
   → lấy mẫu trải đều các tuần + ưu tiên bàn đầy quân, render
   `runs/test_candidates.jpg` (ảnh đánh số) + copy `data/test_candidates/`.
2. Xem ảnh đánh số, chọn ô nào muốn làm test → **báo FEN từng bàn** (như 226–249).
3. Tôi `set_gt.py` + đổi tên (250, 251…) đưa vào `test/bench`, **ĐỒNG THỜI gỡ
   khỏi `incoming/`** để không lọt vào train (⚠️ tránh leakage train/test).
4. Phần còn lại trong `incoming/` → `--merge` vào train như thường.

> ⚠️ **Nguyên tắc vàng:** 1 ảnh chỉ ở MỘT phía — test HOẶC train, không cả hai.
> Ảnh đã chọn làm test phải bị loại khỏi trainset.

---

## Ghi chú
- Pseudo-label chỉ là điểm khởi đầu — gallery review là nơi sửa xe↔mã, tượng↔vua.
- Landmark (board-conner/palace-*) pseudo-label sẵn để train, KHÔNG hiện gallery.
- Re-detect bằng model **mới nhất** mỗi tuần → model càng tốt thì càng ít phải sửa.
- `items_v20/incoming/` không phải split train/val/test → train bỏ qua tới khi `--merge`.
- Retrain: `python scripts/train_items.py --data data/items_v20/data.yaml --name items_vNEXT --img-size 960 --batch-size 12 --workers 2 --no-deploy` → so eval_bench (mốc 229/243) → deploy nếu hơn.
- Lần retrain tới có thể thử **YOLO26** (xem memory) — A/B trước.
