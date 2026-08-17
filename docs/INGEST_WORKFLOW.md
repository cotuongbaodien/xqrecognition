# Ingest workflow — data thật khách gửi → label review → correct → train

> **Thay cho mô hình "weekly"**. Model hiện đã rất ổn (items_v20 deployed), nên
> KHÔNG chạy hàng tuần nữa mà gom **mỗi kỳ 1-2 tháng/lần**. **Mỗi kỳ = 1 folder
> biệt lập** `ingest/<period>/` để review & quyết định độc lập, không trộn kỳ.
>
> Nguyên tắc bất biến:
> 1. **Không để sót** — mọi ảnh *chưa từng vào trainset* đều phải qua quy trình
>    này (kể cả staging kỳ trước còn kẹt).
> 2. **Không leak** — ảnh chọn làm benchmark FEN phải tách ra **trước khi train**,
>    không bao giờ nằm trong `train/`.
> 3. **FEN gate** — deploy model mới CHỈ khi điểm FEN-exact trên bench ≥ model cũ
>    (val mAP KHÔNG dự đoán được FEN — xem `retrain.py`).

---

## Cấu trúc 1 kỳ (gitignored: `/ingest/`)

```
ingest/<period>/                    # vd ingest/2026-07-11
├── inbox/                raw images cần pseudo-label
├── bench_holdout/        ảnh giữ riêng cho benchmark FEN (chờ nhập FEN tay)
├── staging/
│   ├── images/           ảnh đã pseudo-label (pre-review)
│   └── labels/           nhãn YOLO (pseudo)
├── review/
│   ├── <N.tag>/sheet_NNN.jpg   14 gallery/lớp để mắt người soi
│   └── manifest_<tag>.json     map idx → (label file, line)
├── ingest.log
└── STATUS.md             quyết định + tiến độ kỳ này
```

Đích merge chung cho mọi kỳ: `data/items_v20/train/{images,labels}`.

---

## Các bước

### 0. Gom ảnh raw
Bỏ toàn bộ ảnh của kỳ vào `ingest/<period>/inbox/`.
> Nhớ gộp **mọi ảnh chưa train** còn sót (staging kỳ cũ chưa merge → move nhãn
> sẵn có vào `staging/`, ảnh raw vào `inbox/` nếu cần detect lại).

### 1. Tách benchmark holdout (chống leak)
Random ~**350 ảnh** → `bench_holdout/` (ra khỏi `inbox/` để KHÔNG bị train).
Số này sẽ được nhập FEN tay rồi promote vào `test/bench` (bước 8).

### 2. Detect → pseudo-label (chạy ONNX đang deploy)
```bash
python scripts/weekly_ingest.py --batch ingest/<period> \
    --model boarddetection/models/items.onnx
```
- Detect `inbox/` (conf≥0.25, imgsz=960, **device=cpu** vì onnxruntime local chỉ
  có CPUExecutionProvider), ghi nhãn YOLO vào `staging/`, rồi build gallery.
- `.onnx` = đúng model đang chạy prod. Muốn nhanh hơn (GPU) và nhãn tương đương:
  bỏ `--model` để dùng `items.pt` trên GPU (~4× nhanh; cùng weights).

### 3. Review từng lớp (người)
Mở `ingest/<period>/review/<N.tag>/sheet_NNN.jpg`. Mỗi sheet 100 ô (10×10), góc
trên-trái mỗi ô có **số idx (hồng)**. Trong 1 folder, MỌI ô phải đúng quân của
folder đó — ô nào sai thì ghi lại `idx`.

| Quân | Folder | Tag (gõ khi báo) | Hay lẫn |
|---|---|---|---|
| Xe đen 車 | `1.xeden` | `xeden` | mã |
| Mã đen 馬 | `2.maden` | `maden` | xe |
| Xe đỏ 俥 | `3.xedo` | `xedo` | mã |
| Mã đỏ 傌 | `4.mado` | `mado` | xe |
| Pháo đen 包 | `5.phaoden` | `phaoden` | tốt, sĩ |
| Pháo đỏ 炮 | `6.phaodo` | `phaodo` | tốt |
| Tượng đen 象 | `7.tuongden` | `tuongden` | **vua 将** |
| Tượng đỏ 相 | `8.tuongdo` | `tuongdo` | **vua 帥** |
| Tốt đen 卒 | `9.totden` | `totden` | pháo, sĩ |
| Tốt đỏ 兵 | `10.totdo` | `totdo` | pháo |
| Sĩ đen 士 | `11.siden` | `siden` | tượng, tốt |
| Sĩ đỏ 仕 | `12.sido` | `sido` | tượng |
| Vua đen 将 | `13.soaiden` | `soaiden` (hoặc `vua`) | **tượng 象** |
| Vua đỏ 帥 | `14.soaido` | `soaido` (hoặc `vua`) | **tượng 相** |

### 4. Correct — áp sửa
Báo dạng `<idx>=<class> ...`; token class: `xe ma phao si tuong soai/vua tot`
(màu suy từ folder; ép màu: `ma-do`, `xe-den`). `bo`/`del` = xoá box (sentinel 99).
```bash
python scripts/apply_review.py <tag> '5,12=ma 30=phao 41=bo' \
    --dir ingest/<period>/review
```
Idempotent — chỉ đụng ô còn đúng class gốc của gallery. Sửa lại gallery nếu cần:
chạy lại bước 2 (không `--model` nếu chỉ rebuild) — hoặc report tiếp rồi apply.

### 5. Merge vào trainset
```bash
python scripts/weekly_ingest.py --batch ingest/<period> --merge
```
Purge box sentinel-99 (bo) → move `staging/*` vào `data/items_v20/train/`.

### 6. Train A/B — YOLO11 vs YOLO26 (quyết dùng cái nào)
Train 2 lần, **không auto-deploy** (`--no-deploy`), rồi so trên bench:
```bash
# A) YOLO11s (kiến trúc hiện tại)
python scripts/train_items.py --data data/items_v20/data.yaml \
    --name items_vNEXT_y11s --img-size 960 --batch-size 12 --workers 2 \
    --pretrained yolo11s.pt --no-deploy
# B) YOLO26 (thử mới) — cần yolo26s.pt cho A/B công bằng (local mới có yolo26n.pt)
python scripts/train_items.py --data data/items_v20/data.yaml \
    --name items_vNEXT_y26s --img-size 960 --batch-size 12 --workers 2 \
    --pretrained yolo26s.pt --no-deploy
```
> ⚠️ YOLO26 NMS-free: có thể phải chỉnh postprocess trong `onnx_backend.py` khi
> export ONNX — kiểm tra trước khi deploy (xem memory `feedback_yolo11_for_items`).

### 7. Eval + FEN gate → deploy bản thắng
Thêm 2 model vừa train vào dict `MODELS` trong `scripts/eval_bench.py`, rồi:
```bash
python scripts/eval_bench.py          # FEN-exact (mirror-tolerant) + piece-level, bench 243
```
Deploy bản có FEN-exact cao hơn (và ≥ v20):
`cp runs/detect/items_vNEXT_<win>/weights/best.pt boarddetection/models/items.pt`
rồi re-export ONNX (`items.onnx` + `board_seg.onnx`).

### 8. Nhập FEN cho bench_holdout (việc tay, làm nhiều đợt được)

**Chuẩn bị một lần** (đã làm cho kỳ 2026-08-17, lần sau lặp lại y hệt):

1. **Đánh số ảnh tiếp theo bench hiện có** — bench đang có `001..243` thì holdout
   đánh `244..`. Số nhìn thấy lúc nhập chính là số cuối cùng trong `test/bench`,
   khỏi đổi tên lần hai. Sắp theo `created_at` cho tất định.
2. **Xuất FEN nháp từ prod** — `ocr_logs.fen_result` là output của model đang chạy;
   nó KHÔNG phải ground truth nhưng soát nhanh hơn gõ tay rất nhiều (kỳ vừa rồi
   686/700 ảnh có sẵn nháp). Ghi ra `prodfen_draft.txt` dạng `<num>: <fen>`.
3. **Sinh `bench_holdout_mapping.csv`** — `num, uuid, created, huong, trung_train,
   co_fen_nhap` để tra ngược ảnh gốc và biết ảnh nào trùng bàn với train.
4. **Chép ảnh ra một thư mục làm việc** (vd `~/Downloads/bench_holdout/`) để mở xem
   cho tiện; bản gốc 700 ảnh luôn giữ nguyên trong `bench_holdout/`.

**Vòng lặp nhập** — mở `<num>.jpg`, so với dòng `<num>:` trong `prodfen_draft.txt`:

```bash
python scripts/set_holdout_gt.py "<num>: <fen>"   # nháp sai / trống -> FEN của người
python scripts/set_holdout_gt.py "<num> ok"       # nháp đúng -> chép nguyên dòng nháp
python scripts/set_holdout_gt.py "<num> bo"       # ảnh không dùng được -> loại
python scripts/set_holdout_gt.py "244 ok" "245: 3k5/..." "246 bo"   # gộp nhiều cái
```

Script là **nơi duy nhất** ghi `bench_holdout_gt.txt`. Nó validate 10 hàng × 9 cột,
ký tự hợp lệ, và số lượng tối đa từng loại quân trước khi ghi — sai định dạng thì
báo lỗi chứ không ghi bừa. Ảnh đã xong tự bị xoá khỏi thư mục làm việc để khỏi mở
trùng. Dừng giữa chừng lúc nào cũng được, chạy lại là ghi tiếp.

> ⚠️ **Tuyệt đối không đổ `prodfen_draft.txt` thẳng vào `ground_truth.txt`.** Làm vậy
> là để model tự chấm mình 100% → mọi so sánh A/B sau đó vô nghĩa. Chỉ dòng nào
> người đã mắt thấy mới được sang `bench_holdout_gt.txt`.

**Bẫy đã gặp — dán trùng FEN.** Khách hay chụp cùng một thế cờ nhiều lần, nên hai
ảnh khác nhau ra cùng FEN là **bình thường**. Nhưng cũng dễ dán nhầm FEN của ảnh
trước. Cách phân biệt rẻ nhất: so **bản nháp** của hai ảnh — cùng thế thì nháp gần
như trùng (lệch 1-2 ô), dán nhầm thì nháp khác hẳn.
(dHash KHÔNG dùng được ở đây: nó không bất biến với góc chụp, hai ảnh cùng bàn chụp
khác góc vẫn cách nhau cả trăm bit.)

### 9. Promote bench_holdout → test/bench
Khi `bench_holdout_gt.txt` đã đủ số muốn thêm:
```bash
python scripts/set_gt.py <board_id> '<fen>'   # ghi vào test/bench/ground_truth.txt
```
Copy ảnh vào `test/bench/images/`, **xoá khỏi holdout** (đã có GT, không train), rồi
chạy `python scripts/deleak.py --apply` để quarantine ảnh train trùng bàn với bench
mới. Bench lớn hơn = đo FEN tin cậy hơn cho các kỳ sau.

---

## Trạng thái kỳ 2026-07-11
- Nguồn: `download/portal_ocr_2026-07-11` (2883 ảnh thật khách gửi) + zip
  `ocr-batch-...067Z` (30, đã trùng). App-FEN 30 ảnh: `download/portal_ocr_2026-07-11_appfen.json`.
- **bench_holdout: 350** ảnh (chờ nhập FEN).
- **inbox: 2533** ảnh (portal) — detect kỳ này.
- **staging (gộp): 668** ảnh W24-26 kỳ cũ CHƯA train (pseudo-label sẵn) + 2533 mới
  = **~3201** đưa vào 1 review chung.
- Sau review+correct+merge → train A/B YOLO11s vs YOLO26 → eval bench 243 → deploy.
