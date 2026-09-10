# CLAUDE.md — ghi chú cho người/AI làm việc trên repo này

Nhận diện bàn cờ tướng từ ảnh → FEN. Repo track **cả hai**: code train và **code prod
đang chạy thật**. Đọc [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) trước, rồi
[`docs/README.md`](docs/README.md) để biết doc nào còn đúng.

## Chỗ dễ sai nhất — đọc trước khi sửa gì

1. **`boarddetection/` LÀ CODE PROD.** Cây này được copy nguyên lên portal01 và
   bind-mount vào container. Sửa file trong đó = sửa thứ đang phục vụ người dùng thật.
   Deploy: xem `PROD-DEPLOY.md`. Restart `xqdetection` thì **phải restart cả
   `xqdetection-cloudflared`**, không thì tunnel rớt origin (public trả 000/timeout).
2. **`../ocr-gpu-service` là bản mirror CŨ, đừng tin.** Muốn biết prod đang chạy gì thì
   `md5sum` thẳng trên portal01. Bản mirror này từng làm kết luận sai "fix chưa deploy".
3. **Nhánh serving là `refactor/single-items-model`**, không phải `main`/`master`.
4. **Bench `test/bench` (243 ảnh) bị leak** với tập train của items_v20 (156/243). Dùng
   để so A/B thì được; đừng trích số tuyệt đối ra ngoài như độ chính xác thật.
5. **`val mAP` không đáng tin** ở dự án này. Đã có tiền lệ: v10 mAP cao hơn v9 nhưng
   FEN thực tế tệ hơn hẳn. **Luôn chấm bằng FEN trên ảnh thật.**
6. **Đừng sửa nhãn trên Roboflow nữa** — gán nhãn đã chuyển hẳn về local
   (`scripts/weekly_ingest.py` + `apply_review.py`).

## Bố cục

```
boarddetection/       CODE PROD (pipeline, server FastAPI, ONNX backend, models/)
scripts/              CLI rời cho train / eval / review nhãn / cắt video
test/bench/           243 ảnh + ground_truth.txt (FEN người đã audit)
data/items_v20/       tập train hiện tại
ingest/<kỳ>/          ảnh khách gửi theo kỳ + gallery review nhãn
models/backups/       backup model (boarddetection/models/ chỉ giữ bản đang chạy)
docs/                 xem docs/README.md — nhiều doc đã lỗi thời, bảng đó nói rõ cái nào
output/               kết quả chạy (gitignore)
```

## Quy ước code

- Script trong `scripts/` là **standalone**, mở đầu bằng:
  ```python
  ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  sys.path.insert(0, ROOT)
  ```
  argparse dùng docstring làm `description` + `RawDescriptionHelpFormatter`, cờ dài
  kebab-case. Xem `scripts/second_opinion.py` làm mẫu.
- **Đọc/ghi ảnh trên Windows phải qua `imread_u`/`imencode().tofile()`** —
  `cv2.imread` chết với đường dẫn có dấu tiếng Việt, mà data thật toàn tên có dấu.
- Helper FEN dùng chung ở `scripts/_fenutil.py` (`expand_rows`, `mirror_fen`,
  `cell_diff`, `dist_to_start`). **Đừng chép thêm bản thứ tư.**
- Việc nào chạy model tốn thời gian thì **cache ra JSON kèm tham số suy luận**, thêm cờ
  `--repredict` để ép chạy lại (mẫu: `second_opinion.py`, `video_split.py`).
- So FEN phải **chấp nhận soi gương ngang**: pipeline cố ý không chuẩn hoá chiều ngang
  (app phía dưới tự xử lý). Chỉ chuẩn hoá lật dọc 180°.

## Nguyên tắc làm việc đã rút ra từ dự án này

- **Đo trước khi tin.** Nhiều giả thuyết nghe rất hợp lý ở đây đã bị số liệu bác bỏ:
  "bàn lật ngược thiếu trong tập train" (thật ra chiếm 37-44%), "màu là tín hiệu robust
  nhất" (nhầm màu chỉ 1 ô/243 bàn), "board_seg tụt vì config aug" (thật ra vì 30% data
  là bàn rỗng).
- **Sửa nhãn ăn đứt xoá data.** Đã chứng minh: fix 216 nhãn sai → 61/86; xoá data đó đi
  → 52/86.
- **Kết luận cũ bị bác bỏ thì đính chính ngay trong doc cũ**, đừng để hai doc nói ngược
  nhau. `docs/ROT180_TRAINING_GAP.md` là mẫu làm đúng.
- Ghi số liệu phải kèm **bộ dữ liệu + cỡ mẫu**, không viết "tốt hơn" suông.

## Lệnh hay dùng

```bash
python detect.py --image a.jpg --output out/        # 1 ảnh -> FEN
python scripts/eval_bench.py                        # chấm trên bench 243
python scripts/weekly_ingest.py                     # pseudo-label + gallery review nhãn
python scripts/video_split.py <video|thư mục>       # cắt video dài thành clip từng ván
```

Prod cần ffmpeg? **Không** — ffmpeg chỉ dùng cho `video_split.py` ở máy local.
