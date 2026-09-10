# Trạng thái hiện tại

**Cập nhật: 2026-09-10.** Doc này là chỗ bắt đầu — hệ thống đang ở đâu, số đo thật là
bao nhiêu, prod đang chạy gì. Mục lục toàn bộ doc: [`README.md`](README.md).

---

## 1. Hệ thống gồm những gì

```
Ảnh → board_seg.pt (khoanh bàn + cửu cung) → 4 góc → homography → lưới 9×10
                 ↘ items.pt (14 quân + 4 landmark) → snap quân vào lưới → FEN
```

**Chỉ còn 2 model.** Mọi doc nói tới `pieces_det.pt`, `board_det.pt`, `landmarks.pt`
đều là lịch sử — pipeline nhiều model đó đã bỏ từ tháng 5/2026.

| Model | File | Bản | Ngày |
|---|---|---|---|
| Quân + landmark (18 class) | `boarddetection/models/items.pt` (+`.onnx`) | **v20** (YOLO11s @960) | 2026-06-25 |
| Bàn + cửu cung (segmentation) | `boarddetection/models/board_seg.pt` (+`.onnx`) | **v6_synth500** | 2026-06-26 |

Backup nằm ở `models/backups/`; `boarddetection/models/` chỉ giữ bản đang chạy.

## 2. Prod

- Chạy trên **portal01** (`103.175.146.124`), container `xqdetection`, **CPU + ONNX**
  (máy GPU nhà đã bỏ). Public qua Cloudflare tunnel `xqdetection.abcxq.app`.
- Code **bind-mount** → sửa `.py` chỉ cần ship file + restart, không build lại image.
  ⚠ Restart `xqdetection` thì **phải restart cả `xqdetection-cloudflared`**, không thì
  tunnel rớt origin.
- Nhánh serving: `refactor/single-items-model`.
- Chi tiết deploy + env + rollback: [`../PROD-DEPLOY.md`](../PROD-DEPLOY.md).

### Thay đổi mới nhất (2026-09-10, đã deploy)

| Việc | Kết quả đo |
|---|---|
| `OCR_MIN_CONFIDENCE` 0.35 → **0.25** (khớp ngưỡng đã sweep trên bench) | bench 227→229 (bàn thẳng), 219→221 (bàn lật) |
| **Lượt đọc thứ hai**: lượt 1 thiếu tướng thì đọc lại bản xoay 180° (`recognize_image_2pass`) | 600 ảnh prod qua cổng **570 (95,0%) → 582 (97,0%)**; lượt 2 chỉ chạy 1,7% request |
| Tướng nằm ngoài khung thì **snap** về ô hợp lệ chứ không xoá (từ 17/08) | cứu 3/600 ảnh, hỏng 0 |

Rollback không cần build lại: `OCR_TWO_PASS=0` + `OCR_MIN_CONFIDENCE=0.35`.

## 3. Số đo — và cái bẫy phải nhớ

**Bench chính**: `test/bench` — **243 ảnh** kèm FEN ground-truth **đã được người audit**
(lần audit 2026-06-25 phát hiện 9 đáp án GT sai mà model lại đúng).

| Đo trên bench 243 (items v20, conf 0.25, có lượt hai) | Exact FEN |
|---|---|
| Ảnh chụp thẳng | **232/243** |
| Ảnh bàn lật (xoay 180°) | **223/243** |

> ⚠ **Số tuyệt đối bị phồng.** Đo dHash thấy **156/243 bàn trong bench trùng với tập
> train của items_v20**. Đã de-leak tập train (`scripts/deleak.py`, cách ly 307 ảnh) nhưng
> **v20 được train TRƯỚC khi de-leak** → bench vẫn không phải held-out thật với model này.
> Dùng bench để **so A/B giữa hai cấu hình** thì vẫn đúng; đừng trích con số này ra ngoài
> như độ chính xác thật.
>
> Bộ held-out sạch: `ingest/2026-07-11/bench_holdout/` (700 ảnh, cân bằng hướng
> 343 đỏ-dưới / 307 đen-dưới / 50 khó) — **mới nhập GT 60/700**, chưa dùng chấm được.

**Lỗi còn lại là gì** (đo trên bench 243, ảnh chụp thẳng):

| Loại lỗi | Số ô |
|---|---|
| Nhầm LOẠI quân — gần hết là **xe ↔ mã** | 38 (35 là xe/mã) |
| Sót quân | 28 |
| Thừa quân | 14 |
| **Nhầm MÀU đen ↔ đỏ** | **1** |

⇒ Đừng đầu tư vào "tách màu đen/đỏ", nó không hỏng. Nút thắt là **xe↔mã và sót quân**.
Tiền xử lý ảnh (tăng bão hoà, CLAHE, imgsz 1280) đã thử: **đều tệ hơn**.

**Trên ảnh prod thật** (600 ảnh lấy ngẫu nhiên từ kỳ 08): sau thay đổi 10/09, **97,0%**
qua cổng "có đủ 2 tướng + ≥5 quân" (tức prod trả `detected=true`).

## 4. Việc đang mở

| Việc | Trạng thái |
|---|---|
| **Review nhãn kỳ 2026-07-11** (bộ "chỉ ô nghi ngờ", 252 sheet ≈ 7,4 giờ) | Đang làm dở. Đây là đường chính để hạ lỗi xe↔mã |
| Train lại items sau khi review xong | Chưa. Lần tới thử **YOLO26**, bắt buộc A/B với YOLO11 trên bench 243 |
| Nhập FEN GT cho `bench_holdout` 700 ảnh | Mới 60/700 |
| Sinh dữ liệu rot180 | **Đã bị số liệu bác bỏ là ưu tiên** — xem ROT180_TRAINING_GAP |
| Đa dạng skin bàn (bàn gỗ quân đỏ không có mực đỏ) | Chưa làm, đây mới là lỗ hổng đáng đầu tư |

## 5. Công cụ hay dùng

| Lệnh | Việc |
|---|---|
| `python detect.py --image a.jpg --output out/` | Đọc 1 ảnh / cả thư mục ra FEN |
| `python scripts/eval_bench.py` | Chấm model trên bench 243 |
| `python scripts/weekly_ingest.py` | Pseudo-label + dựng gallery review nhãn |
| `python scripts/second_opinion.py` | Lọc ô đáng ngờ (soi 20% số ô, bắt ~99% lỗi) |
| `python scripts/video_split.py <video>` | Cắt video dài thành clip từng ván ([doc](VIDEO_SPLIT.md)) |
