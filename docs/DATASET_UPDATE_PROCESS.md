# Dataset Update Process

Mỗi khi có dataset mới từ Roboflow (vd `itemdetectionYYMMDD.yolov8.zip`), follow process này.

## Quick command (1 dòng)

```bash
python scripts/retrain.py --zip data/itemdetectionYYMMDD.yolov8.zip --name itemsYYMMDD
```

Script tự động: extract → split 80/15/5 → train → backup model cũ → deploy model mới → test.

## Manual steps (nếu cần debug từng bước)

### Step 1: Extract zip vào folder mới

```bash
python -c "
import zipfile
from pathlib import Path
zip_path = 'data/itemdetectionYYMMDD.yolov8.zip'
dest = Path('data/items_vN')  # tăng N mỗi lần
dest.mkdir(exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(dest)
"
```

Hoặc PowerShell:
```powershell
Expand-Archive data/itemdetectionYYMMDD.yolov8.zip data/items_vN
```

### Step 2: Verify dataset

```bash
# Check counts per class — palace-bottom thường yếu, cần ≥4/ảnh
python -c "
from pathlib import Path
labels = Path('data/items_vN/train/labels').glob('*.txt')
from collections import Counter
c = Counter()
for f in labels:
    for line in f.read_text().splitlines():
        if line.strip():
            c[int(line.split()[0])] += 1
NAMES = 'black-advisor black-cannon black-chariot black-elephant black-general black-horse black-soldier board-conner palace-bottom palace-center palace-conner red-advisor red-cannon red-chariot red-elephant red-general red-horse red-soldier'.split()
for i, n in enumerate(NAMES):
    print(f'{i:2} {n:20} {c.get(i, 0)}')
"
```

Check:
- Total classes = 18
- Pieces (0-6, 11-17) đều có instances
- **Landmarks** (7-10): `board-conner` ~3-4/ảnh, `palace-bottom` ~4/ảnh, `palace-center` ~2/ảnh, `palace-conner` ~4/ảnh

### Step 3: Split 80/15/5

```bash
python scripts/split_items.py --dir data/items_vN
```

### Step 4: Backup current model

```bash
cp boarddetection/models/items.pt models/backups/items_v{N-1}.pt
```

### Step 5: Train với augmentation đầy đủ

```bash
python scripts/train_items.py \
  --data data/items_vN/data.yaml \
  --pretrained yolov8s.pt \
  --epochs 200 \
  --batch-size 16 \
  --device cuda \
  --seed 42 \
  --name items_vN
```

Train tự động:
- `degrees=45` — rotation augment ±45°
- `perspective=0.0015` — phối cảnh
- `fliplr=0.5` — flip ngang (board đối xứng trái-phải)
- `scale=0.5`, `mosaic=1.0`, `mixup=0.1`, `copy_paste=0.1`

Output: `boarddetection/models/items.pt` (active) + `boarddetection/models/items_items_vN.pt` (backup)

### Step 6: Test trên test set

```bash
python detect.py --dir test --output test/output --confidence 0.3
```

So sánh FEN với `test/baseline_fen.md`. Update file đó với kết quả mới.

### Step 7: Nếu kết quả tệ hơn → rollback

```bash
cp models/backups/items_v{N-1}.pt boarddetection/models/items.pt
```

## Khi nào RETRAIN

Retrain MỖI khi:
- Có ảnh mới được label
- Sửa label cũ
- Đổi augmentation params

KHÔNG cần retrain khi:
- Sửa code post-processing (`pipeline.py`, `item_detector.py`)
- Sửa `rules_validator.py`
- Sửa visualization

## Augmentation params

Trong `scripts/train_items.py`:

| Param | Hiện tại | Khi nào tăng |
|---|---|---|
| `degrees` | 45 | Tăng lên 90 nếu user chụp landscape nhiều |
| `perspective` | 0.0015 | Tăng nếu nhiều ảnh có camera nghiêng |
| `fliplr` | 0.5 | Giữ — board đối xứng |
| `flipud` | 0.0 | **GIỮ 0** — không flip dọc (chữ Hán bị ngược) |
| `mosaic` | 1.0 | Giữ — tốt cho dataset nhỏ |
| `mixup` | 0.1 | Có thể tăng 0.2 nếu overfit |
| `copy_paste` | 0.1 | Có thể tăng nếu thiếu pieces trong class |

## Labeling checklist (cho mỗi ảnh mới trong Roboflow)

**v6+: 19 classes** (thêm `board-border`)

Bắt buộc label đầy đủ:

1. **Pieces** (14 classes): mọi quân cờ trên board
2. **board-conner** (4 điểm): 4 góc bàn cờ (label đè lên quân nếu bị che)
3. **palace-center** (2 điểm): tâm X mỗi palace
4. **palace-conner** (4 điểm): 2 góc trong mỗi palace × 2 palace
5. **palace-bottom** (4 điểm): 2 đỉnh palace ở back rank × 2 palace
6. **board-border** (26 điểm) — **MỚI**: các điểm grid intersection trên perimeter
   - Top edge: cols 1, 2, 4, 6, 7 (5 điểm)
   - Bottom edge: cols 1, 2, 4, 6, 7 (5 điểm)
   - Left edge: rows 1-8 (8 điểm)
   - Right edge: rows 1-8 (8 điểm)

**Tổng landmarks: 40 điểm/ảnh.** Label đè lên quân cờ nếu bị che.

**Why board-border**: 34 perimeter intersections (board-conner + palace-bottom + board-border). Khi mid-game 10-15 điểm bị che, vẫn còn ~20+ điểm để fit chính xác board outline → solve orientation/rotation issue.
