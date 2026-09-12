# Mục lục tài liệu

Repo này có nhiều doc viết ở nhiều thời điểm, **không phải cái nào cũng còn đúng**.
Bảng dưới nói rõ cái nào đang dùng, cái nào chỉ còn giá trị lịch sử.

Bắt đầu từ đâu: [`CURRENT_STATUS.md`](CURRENT_STATUS.md) (hệ thống đang ở đâu) →
[`../PROD-DEPLOY.md`](../PROD-DEPLOY.md) (prod chạy thế nào) →
[`INGEST_WORKFLOW.md`](INGEST_WORKFLOW.md) (vòng cải thiện model).

---

## Đang dùng — đọc mấy cái này

| Doc | Nội dung | Cập nhật |
|---|---|---|
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | **Bắt đầu ở đây.** Model nào đang chạy, số đo thật, prod đang ở trạng thái nào | 2026-09-10 |
| [`../PROD-DEPLOY.md`](../PROD-DEPLOY.md) | Prod chạy ở đâu, deploy ra sao, env chỉnh chất lượng đọc, cách rollback | 2026-09-10 |
| [`VIDEO_SPLIT.md`](VIDEO_SPLIT.md) | Cắt video dài thành clip từng ván (`scripts/video_split.py`) | 2026-09-10 |
| [`VIDEO_BATCH_RUNBOOK.md`](VIDEO_BATCH_RUNBOOK.md) | **Chạy tiếp lô cắt kho `E:/videos/GiangHo`**: trạng thái (768 video / 3 996 ván), lệnh, dừng sạch, 3 luật phải nhớ, ván dài, quy ước tên `YYYYMMDD_` | 2026-09-12 |
| [`VIDEO_SOURCES.md`](VIDEO_SOURCES.md) | **Danh sách mọi nguồn video + thứ tự làm**: 515 file cục bộ + 1028 video YouTube (713 ẩn), bài toán chỗ chứa, cách lấy cookie | 2026-09-11 |
| [`ROT180_TRAINING_GAP.md`](ROT180_TRAINING_GAP.md) | Bàn lật ngược: giả thuyết cũ đã bị số liệu bác bỏ, số đo tách hướng, lượt đọc thứ hai | 2026-09-10 |
| [`INGEST_WORKFLOW.md`](INGEST_WORKFLOW.md) | Ảnh khách gửi → pseudo-label → gallery review → merge → train. **Quy trình chính để cải thiện model** | 2026-08-17 |
| [`xqdetection-vps-cpu.md`](xqdetection-vps-cpu.md) | Runbook VPS CPU/ONNX: đổi code, đổi model, xem log, chạy ingest tay | 2026-07-11 |
| [`onnx-cpu-vps-handoff.md`](onnx-cpu-vps-handoff.md) | Vì sao và bằng cách nào dời OCR từ GPU nhà sang VPS CPU + ONNX | 2026-07-11 |
| [`ALGORITHM_AUDIT.md`](ALGORITHM_AUDIT.md) | Audit giải thuật (33 finding đã verify) — chỗ nào trong pipeline còn yếu | 2026-06-16 |
| [`WEEKLY_INGEST.md`](WEEKLY_INGEST.md) | Chi tiết `scripts/weekly_ingest.py` (công cụ, không phải nhịp làm việc — nhịp xem INGEST_WORKFLOW) | 2026-06-26 |
| [`DATASET_GUIDE.md`](DATASET_GUIDE.md) | Tên class chuẩn + cách chuẩn bị dataset mới | — |

## Còn dùng khi cần, phạm vi hẹp

| Doc | Nội dung |
|---|---|
| [`BENCH_AUDIT.md`](BENCH_AUDIT.md) | Lần audit ground-truth bench 224 ảnh (phát hiện GT sai) — bối cảnh cho bench hiện tại |
| [`SESSION_LOG_2026-06-16.md`](SESSION_LOG_2026-06-16.md) | Nhật ký deploy board_seg_v5 + palace |
| [`SYNTHETIC_DATA_PLAN.md`](SYNTHETIC_DATA_PLAN.md), [`SYNTH_V3_PLAN.md`](SYNTH_V3_PLAN.md), [`SYNTH_ASSETS_REPORT.md`](SYNTH_ASSETS_REPORT.md) | Sinh ảnh tổng hợp: kế hoạch, phân tích vì sao synth v1 tệ hơn, kho asset |
| [`DATASET_UPDATE_PROCESS.md`](DATASET_UPDATE_PROCESS.md) | Quy trình nhận dataset mới từ Roboflow (**lưu ý**: gán nhãn giờ làm local, không qua Roboflow nữa) |
| [`RETRAIN_GUIDE.md`](RETRAIN_GUIDE.md) | Hướng dẫn retrain — viết cho `pieces_det.pt` cũ, nguyên tắc vẫn dùng được |
| [`USAGE.md`](USAGE.md) | Hướng dẫn dùng chi tiết — một phần API đã đổi, xem README gốc trước |

## Lịch sử — KHÔNG còn mô tả hệ thống hiện tại

Giữ lại vì ghi được vì sao ngày xưa quyết định như vậy. Đừng dựa vào để làm việc hôm nay.

| Doc | Vì sao lỗi thời |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md), [`ACCURACY_ANALYSIS.md`](ACCURACY_ANALYSIS.md) | Mô tả pipeline nhiều model (`pieces_det.pt` + `board_det.pt` + `landmarks.pt`). Nay chỉ còn **items.pt + board_seg.pt** |
| [`DOMAIN_GAP_ANALYSIS.md`](DOMAIN_GAP_ANALYSIS.md), [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md), [`NEXT_STEPS.md`](NEXT_STEPS.md), [`TRAINING_RESULTS.md`](TRAINING_RESULTS.md) | Số liệu và kế hoạch của giai đoạn 04-05/2026, đã bị các vòng train sau vượt xa |
| [`AUTO_LEARN_SOLUTION.md`](AUTO_LEARN_SOLUTION.md) | Ý tưởng auto-learn ban đầu; thực tế nay là INGEST_WORKFLOW |
| [`CLEANUP.md`](CLEANUP.md), [`DATA_AUDIT.md`](DATA_AUDIT.md) | Nói về cây thư mục `src/` và dataset đã không còn |
| [`ORIENTATION_FIX.md`](ORIENTATION_FIX.md) | Phần phân tích vẫn đúng, nhưng kết luận "màu là tín hiệu robust nhất" **đã bị số liệu bác bỏ** — xem ROT180_TRAINING_GAP |

---

## Quy ước khi viết doc mới

- **Ghi ngày** ngay đầu doc, và ghi rõ *đo được* hay *phỏng đoán*.
- Kết luận cũ bị bác bỏ thì **đính chính ngay trong doc cũ** (gạch ngang + trỏ sang doc
  mới), đừng để hai chỗ nói ngược nhau — `ROT180_TRAINING_GAP.md` là mẫu.
- Số liệu phải kèm **bộ dữ liệu và cỡ mẫu** ("bench 243", "600 ảnh prod"), không viết
  chung chung "tốt hơn".
- Doc mới phải thêm một dòng vào bảng ở trên.
