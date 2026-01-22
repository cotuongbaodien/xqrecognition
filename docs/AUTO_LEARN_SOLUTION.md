# Auto-Learn Pipeline - Solution Chi Tiết

> **TRẠNG THÁI:** Chờ FEN data (848 images đã có)
>
> **TRAINING METHOD:** Fine-tune từ `pieces_det.pt`
>
> **PIPELINE:** Approach 1 → Approach 2 (theo thứ tự)
>
> **NEXT:** Khi FEN xong → nói "chạy đi" → Claude sẽ tự động chạy pipeline

---

## 1. Quyết Định Đã Chọn

### 1.1 Training Method: Fine-tune

```
pieces_det.pt (model hiện có)
         │
         ▼
    Fine-tune với dataset mới
         │
         ▼
    pieces_det_v2.pt
```

### 1.2 Pipeline: 2 Approaches Theo Thứ Tự

```
┌─────────────────────────────────────────────────────────────────┐
│  APPROACH 1: Generate Annotations → Train                        │
│  (Làm trước)                                                     │
│                                                                  │
│  FEN + board_det.pt → YOLO labels → Train → Model v1            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  APPROACH 2: Semi-Supervised Refinement                          │
│  (Làm sau để cải thiện)                                          │
│                                                                  │
│  Model v1 predictions + FEN labels → Compare → Refine           │
│  → Train → Model v2 (tốt hơn)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Data Sources

- Screenshots từ app cờ
- Ảnh chụp bàn cờ thật
- Frames từ video YouTube

---

## 2. Cấu Trúc Data

### 2.1 File Structure

```
data/
└── prepare/
    ├── images/              ← Bạn thêm ảnh vào đây
    │   ├── 001.png
    │   ├── 002.png
    │   └── ... (tối đa 999 ảnh)
    ├── labels.csv           ← Đã có 999 dòng, bạn điền FEN
    └── README.md
```

### 2.2 labels.csv Format

```csv
image,fen
001.png,rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
002.png,r1bakab1r/9/1cn3nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
003.png,
...
999.png,
```

### 2.3 FEN Ký Hiệu

| Quân | Đen | Đỏ |
|------|-----|-----|
| Xe | r | R |
| Mã | n | N |
| Tượng | b | B |
| Sĩ | a | A |
| Tướng | k | K |
| Pháo | c | C |
| Tốt | p | P |

---

## 3. APPROACH 1: Generate Annotations → Train

### 3.1 Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: BẠN CHUẨN BỊ DATA                                       │
│  - Thêm ảnh vào data/prepare/images/                            │
│  - Điền FEN vào data/prepare/labels.csv                         │
│  - Chạy: python scripts/auto_learn/validate.py                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: AUTO GENERATE ANNOTATIONS                               │
│  - Detect board bbox (dùng board_det.pt)                        │
│  - Build grid 9x10                                              │
│  - Parse FEN → piece positions                                  │
│  - Generate YOLO bounding boxes                                 │
│  - Export labels/*.txt                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: TRAIN                                                   │
│  - Split: 80% train, 10% val, 10% test                         │
│  - Fine-tune pieces_det.pt                                      │
│  - 30-50 epochs                                                 │
│  - Output: Model v1 (pieces_det_v1.pt)                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Chi Tiết Annotation Generation

```
Image + FEN
     │
     ▼
┌─────────────────┐
│ 1. Load Image   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Detect Board │ ← board_det.pt
│    (get bbox)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Build Grid   │ ← Chia bbox thành 9x10 = 90 điểm
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Parse FEN    │ ← "rnbakabnr/..." → [(0,0,'r'), (0,1,'n'), ...]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Generate     │ ← Mỗi quân → bbox tại grid cell
│    Bboxes       │    size = cell_size × 0.8
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Export YOLO  │ ← "class_id x_center y_center width height"
│    Label (.txt) │
└─────────────────┘
```

### 3.3 Output Approach 1

- `models/pieces_det_v1.pt` - Model đã fine-tune
- Accuracy estimate: ~85-90%

---

## 4. APPROACH 2: Semi-Supervised Refinement

> **Khi nào làm:** Sau khi hoàn thành Approach 1, nếu muốn cải thiện accuracy

### 4.1 Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: CHẠY MODEL V1 TRÊN DATASET                              │
│  - Dùng Model v1 detect pieces trên tất cả images               │
│  - Lưu predictions                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: COMPARE & VALIDATE                                      │
│                                                                  │
│  FEN-generated labels    vs    Model v1 predictions             │
│         │                              │                         │
│         └──────────┬───────────────────┘                         │
│                    │                                             │
│                    ▼                                             │
│              ┌───────────┐                                       │
│              │  Compare  │                                       │
│              └─────┬─────┘                                       │
│                    │                                             │
│         ┌─────────┼─────────┐                                   │
│         ▼         ▼         ▼                                   │
│      Match    FEN-only   Det-only                               │
│      (Good)   (Missing)  (False+)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: REFINE LABELS                                           │
│  - Keep matches (high confidence)                               │
│  - Review FEN-only (model missed) → add to training             │
│  - Review Det-only (false positives) → remove or fix            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: RETRAIN                                                 │
│  - Train với refined labels                                     │
│  - Output: Model v2 (pieces_det_v2.pt)                          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Output Approach 2

- `models/pieces_det_v2.pt` - Model cải thiện
- Accuracy estimate: ~90-95%

---

## 5. Scripts

### 5.1 Đã Tạo

```
scripts/auto_learn/
├── __init__.py
└── validate.py      ✅ Validate data
```

### 5.2 Sẽ Tạo (Approach 1)

```
scripts/auto_learn/
├── fen_parser.py         ⏳ Parse FEN → positions
├── annotation_gen.py     ⏳ Generate YOLO labels
├── dataset_builder.py    ⏳ Split train/val/test
└── train.py              ⏳ Fine-tune training
```

### 5.3 Sẽ Tạo (Approach 2)

```
scripts/auto_learn/
├── predict.py            ⏳ Run Model v1 predictions
├── compare.py            ⏳ Compare predictions vs FEN
└── refine.py             ⏳ Refine labels
```

---

## 6. Commands

### Approach 1

```bash
# 1. Validate data
python scripts/auto_learn/validate.py

# 2. Generate YOLO annotations
python -m scripts.auto_learn.generate

# 3. Build dataset
python -m scripts.auto_learn.build

# 4. Train Model v1
python -m scripts.auto_learn.train --epochs 50
```

### Approach 2

```bash
# 5. Run predictions với Model v1
python -m scripts.auto_learn.predict

# 6. Compare và refine labels
python -m scripts.auto_learn.compare

# 7. Train Model v2
python -m scripts.auto_learn.train --epochs 30 --resume
```

---

## 7. Checklist

### Bạn cần làm (Hiện tại):
- [ ] Thêm ảnh vào `data/prepare/images/`
- [ ] Điền FEN vào `data/prepare/labels.csv`
- [ ] Chạy validate để kiểm tra
- [ ] Báo khi sẵn sàng

### Tôi sẽ làm (Approach 1):
- [ ] Implement annotation generator
- [ ] Implement dataset builder
- [ ] Implement training script
- [ ] Train Model v1

### Sau đó (Approach 2):
- [ ] Implement comparison tools
- [ ] Refine labels
- [ ] Train Model v2

---

## 8. Timeline

| Phase | Task | Status |
|-------|------|--------|
| Setup | Folder structure | ✅ Done |
| Setup | labels.csv template (999 rows) | ✅ Done |
| Setup | validate.py | ✅ Done |
| **Data** | **Chuẩn bị images + FEN** | **⏳ Bạn đang làm** |
| Approach 1 | Implement scripts | 🔜 Chờ data |
| Approach 1 | Train Model v1 | 🔜 |
| Approach 2 | Implement refinement | 🔜 Sau Model v1 |
| Approach 2 | Train Model v2 | 🔜 |

---

## 9. Notes

- **Minimum data:** 100 images (recommend 500+)
- **Approach 1 training:** ~1-2 giờ (GPU) / ~4-6 giờ (CPU)
- **Approach 2:** Optional, chỉ làm nếu muốn cải thiện

Ping tôi khi data sẵn sàng!
