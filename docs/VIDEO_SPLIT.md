# Cắt video cờ tướng thành clip từng ván (`scripts/video_split.py`)

Đưa vào một video dài (livestream cờ giang hồ 1-2 giờ, nhiều ván nối nhau) →
tự tìm mốc bắt đầu từng ván → cắt ra nhiều clip + xuất chuỗi FEN theo thời gian.

Dùng đúng detector đang chạy prod (`items.pt` + `board_seg.pt`), **không đụng gì
trong `boarddetection/`** nên không ảnh hưởng service.

---

## 1. Ý tưởng

Bàn cờ vừa được xếp lại về **thế khai cuộc** chính là mốc bắt đầu ván mới. Detector
đọc được thế cờ trên từng frame, nên chỉ cần đo:

```
d = số ô lệch giữa thế đọc được và thế khai cuộc   (0 → 90)
```

Đo trên video thật (`Nhâm Thầy Cúng vs Nghĩa Sing.mp4`, 60 phút, bàn gỗ, camera
điện thoại gần như cố định):

| Tình huống | `d` | số quân |
|---|---|---|
| Giữa/tàn ván | **31-40** | 13-24 |
| Đang xếp lại bàn | 39 → 8 (giảm dần) | 22 → 29 (tăng dần) |
| **Vừa xếp xong = mốc ván** | **0-2** | **30-32** |
| Đã đi vài nước | 3-12 | 29-32 |

Khoảng trống giữa "khai cuộc" (0-9) và "không khai cuộc" (31-40) rộng **22 ô** —
đây là lý do phương pháp này chắc ăn, không phải nhờ ngưỡng khéo.

### Chữ ký một mốc ván (quét 1 fps quanh 19:00)

```
17:30-18:55   22 quân, d≈39-41     ← tàn cuộc ván trước
18:55-19:08   22→29 quân, d 39→8   ← ĐANG XẾP LẠI (tay che, đọc rất nhiễu)
19:09-19:17   32 quân, d=0          ← MỐC BẮT ĐẦU VÁN
19:18+        29-32 quân, d=4,5,9…  ← đã đi nước
```

Hai cái bẫy đã đo được, luật nhận mốc phải sống với chúng:

1. **Độ dài pha `d=0` không cố định.** Mốc 19:09 giữ `d=0` suốt **9 giây**; mốc
   53:02 chỉ `d=0` **đúng 1 mẫu** rồi hai bên đi ngay. Luật "phải giữ ≥3 mẫu liên
   tiếp" nghe hợp lý nhưng **bỏ sót ván thứ 6**.
2. **Pha xếp bàn nhiễu nặng.** Chuỗi `d` thật lúc 52:51→53:02:
   `4, 3, 23, 17, 8, 2, 23, 12, 3, 0` — tay che làm nhảy loạn. Vì vậy phải gộp cụm
   rồi lấy **mẫu CUỐI**, không lấy mẫu đầu.

---

## 2. Chạy

```bash
# đủ 3 bước: quét → tìm mốc → cắt
python scripts/video_split.py "E:\videos\GiangHo\ten video.mp4"

# chỉ quét + tìm mốc (KHÔNG cắt) — xem starts.jpg trước rồi mới quyết
python scripts/video_split.py <video> --stage scan --stage segment

# sửa ngưỡng, tính lại mốc: KHÔNG nạp model, ~1 giây
python scripts/video_split.py <video> --stage segment --start-dist 3 --min-gap 90

# xem lệnh ffmpeg sẽ chạy mà chưa cắt
python scripts/video_split.py <video> --stage cut --dry-run

# cắt thật
python scripts/video_split.py <video> --stage cut
```

Cần `ffmpeg` + `ffprobe` trong PATH (phụ thuộc ngoài, không phải lib python).

### Thư mục kết quả — nằm NGAY CẠNH video gốc

Mỗi video một thư mục tự chứa đủ, đặt cùng chỗ với video (không phải trong repo):

```
E:\videos\GiangHo\Nhâm Thầy Cúng vs Nghĩa Sing\
    00_goc_Nhâm Thầy Cúng vs Nghĩa Sing.mp4     <- VIDEO GỐC, đã được dời vào
    Nhâm Thầy Cúng vs Nghĩa Sing_van01_00-00-09.mp4
    Nhâm Thầy Cúng vs Nghĩa Sing_van02_00-11-23.mp4
    …
    starts.jpg      <- ảnh mốc từng ván, soi trước khi tin
    index.csv       <- mốc yêu cầu / mốc thật sau snap / độ dài / dung lượng
    _data\          <- scan.json, games.json, rejected.json, timeline.csv, fens\
```

| File | Nội dung |
|---|---|
| `00_goc_<tên>.mp4` | video gốc. Tiền tố `00_goc_` để phân biệt hẳn với clip và luôn nằm đầu khi sắp theo tên. `--no-move` nếu muốn để video ở chỗ cũ |
| `<tên>_vanNN_<hh-mm-ss>.mp4` | clip từng ván; tên mang theo tên video nên tách khỏi thư mục vẫn biết của video nào |
| `starts.jpg` | **ảnh dán frame lúc bắt đầu mỗi ván** — cổng kiểm tra bằng mắt trước khi cắt |
| `index.csv` | mốc yêu cầu / mốc thật sau snap keyframe / độ dài / dung lượng |
| `_data/scan.json` | tham số + **mọi mẫu đã đọc** (`t`, `fen`, số quân, `d`, `gate`, `conf`). Cache: chạy lại không detect lại |
| `_data/games.json` | mốc bắt đầu/kết thúc từng ván + thống kê chuỗi FEN |
| `_data/rejected.json` | ứng viên **bị loại** kèm lý do — để biết vì sao một mốc không được nhận |
| `_data/timeline.csv` | mỗi mẫu một dòng, mở bằng Excel để tự soi |
| `_data/fens/gNN_fens.jsonl` | chuỗi quan sát FEN theo thời gian của từng ván |

Chạy lại lần sau cứ đưa **đường dẫn video cũ** — script tự tìm bản đã dời vào
(`00_goc_…`) và dùng tiếp. Dời video trong cùng ổ đĩa chỉ là đổi tên nên tức thì;
khác ổ thì phải copy cả GB và script báo trước. `--out DIR` để tự chọn chỗ khác.

---

## 3. Ba stage

```
scan     decode + detect   (phần đắt duy nhất, có cache)
segment  logic thuần       (<1 giây, chỉnh ngưỡng thoải mái)
cut      ffmpeg -c copy    (vài giây cho cả video)
```

### 3.1 `scan`

- **Decode bằng MỘT tiến trình ffmpeg** (`-vf fps=1/20`). Cả video 1 giờ mất **28 s**.
  Tuyệt đối không `-ss` từng mốc: mỗi seek ~0,5-1 s, chậm gấp hàng trăm lần.
- **Detect** bằng `recognize_image_2pass(conf=0.25)` — đúng đường prod, gồm cả lượt
  đọc lại bản xoay 180°. 181 frame mất **16 s** trên GPU (~40 s CPU/ONNX).
- **Quét tinh**: mỗi ứng viên mở một cửa sổ `[t-60s, t+30s]` quét lại ở 1 fps để chốt
  đúng giây.
- **Cache**: `scan.json` lưu cả tham số lẫn kết quả; đổi `--step`/`--conf` thì tự quét
  lại, còn `--repredict` là ép chạy lại từ đầu.

> Mỗi lượt decode ghi vào thư mục con riêng (`frames/c/`, `frames/w01/`…). Dùng chung
> một thư mục thì frame lượt trước còn lại làm **lệch toàn bộ timestamp** — bug này đã
> dính một lần khi phát triển, mốc ván lệch 3-6 phút mà nhìn không ra.

### 3.2 `segment` — luật nhận mốc

1. **Mẫu khai cuộc** = `d <= --start-dist` (2) **và** `>= --start-pieces` (30) quân.
2. **Gộp cụm**: các mẫu khai cuộc cách nhau `<= --cluster-gap` (30 s) là một cụm.
3. **Mốc ván = mẫu CUỐI của cụm** (sau đó `d` chỉ tăng vì đã đi nước).
4. **Xác nhận ván trước đã tàn**: trong `--confirm-window` (60 s) trước cụm phải có
   `>= --confirm-min` (3) mẫu với `d >= 25` hoặc `<= 25` quân. Ván đầu video được miễn.
5. **Debounce**: hai mốc cách nhau `< --min-gap` (120 s) thì bỏ mốc sau.

Ứng viên bị loại ở bước 4/5 **được ghi vào `rejected.json`** kèm lý do, không im lặng.

**Kết thúc ván N** = mốc ván N+1 trừ `--reset-lead` (25 s); ván cuối kéo tới hết video.
Không đi dò "bàn bị dọn" vì pha đó là đoạn nhiễu nhất — mốc ván sau đã đủ để suy ra.

### 3.3 `cut`

```
ffmpeg -ss <start> -i <video> -t <dur> -map 0:v:0 -map 0:a? \
       -c copy -avoid_negative_ts make_zero -movflags +faststart out.mp4
```

- `-ss` **trước** `-i`: seek theo index, không decode. Với `-c copy` ffmpeg lùi về
  keyframe gần nhất → chỉ **dư ở đầu, không bao giờ cụt**. Video mẫu có keyframe mỗi
  6,0 s nên lệch tối đa 6 s; `index.csv` in cả mốc yêu cầu lẫn **mốc thật sau snap**.
- `-map 0:v:0 -map 0:a?` chứ không `-map 0`: `-map 0` kéo theo cả stream dữ liệu mp4
  mà copy sang mp4 hay lỗi; dấu `?` để video không có tiếng cũng chạy.
- **Mặc định lấy dư `--lead 300` giây TRƯỚC và `--tail 300` giây SAU mỗi ván** để chắc
  chắn không sót. Hệ quả: các clip **chồng lấn nhau** và tổng dung lượng lớn hơn video
  gốc (video mẫu 0,72 GB → 6 clip tổng 1,3 GB). Muốn gọn thì hạ xuống `--lead 20 --tail 20`.
- Cắt xong thì **dời luôn video gốc vào thư mục** thành `00_goc_<tên>.<ext>` (`--no-move`
  để tắt). Chỉ dời khi đã cắt được ít nhất một clip, và không dời khi `--dry-run`.
- Clip đã tồn tại thì bỏ qua (chạy lại được); `--overwrite` để ghi đè.

---

## 4. Chuỗi FEN theo thời gian

`fens/gNN_fens.jsonl`, dòng đầu là `_meta`, mỗi dòng sau là một **thế đã lọc**:

```jsonc
{"t": 3196.0, "t_last": 3198.0, "fen": "r1bakabnr/9/...", "n": 32,
 "conf": 0.894, "seen": 3, "d_prev": 2}
```

- `seen` = thế đó xuất hiện ở mấy mẫu liên tiếp (`--min-seen`, mặc định 2) — lọc nhiễu
  một-frame. Cần thiết vì đo được **110/180 frame liên tiếp đổi FEN** giữa ván do tay
  che + lỗi detect.
- `t` / `t_last` = lần đầu / lần cuối thấy thế đó → xấp xỉ thời gian suy nghĩ.
- `d_prev` = số ô đổi so với thế đã ghi trước đó. **Một nước cờ hợp lệ đổi ĐÚNG 2 ô**
  (ô đi khỏi thành trống, ô đến đổi quân — ăn quân cũng vậy). Tỉ lệ `d_prev == 2` chính
  là thước đo chuỗi này sạch tới đâu, in ra trong `_meta.thong_ke`.

### ⚠ Đây KHÔNG phải biên bản nước đi

Nó là **chuỗi quan sát thô của detector**, chưa hề kiểm tra luật cờ. Ở `--fen-step 5`
(mặc định) chỉ khoảng 25-40% bước chuyển là `d_prev == 2`; phần còn lại là do lấy mẫu
nhảy qua 2 nước (`d_prev == 4`) hoặc đọc sai lúc tay che.

Muốn ra biên bản thật cần một bước riêng, chưa làm:
1. Quét dày hơn (`--fen-step 1`) để không nhảy nước.
2. Lọc theo luật: chỉ nhận bước chuyển đúng một nước hợp lệ; chỗ nhảy cóc thì quét lại
   dày hơn quanh mốc đó.
3. Theo dõi lượt đi, seed từ việc bên nào đi trước ở thế khai cuộc.
4. Vá đoạn tay che bằng cách nội suy giữa hai thế tin cậy hai bên.

---

## 5. ROI bàn cờ

- ROI = **median** bbox quad của `BoardSegmenter` trên ~20 frame rải đều, đệm 10%.
  Bắt buộc dùng median: đo trên video dọc 720x1280, mép trên/dưới rất ổn (y 516→912
  suốt 10 phút) nhưng **mép trái nhảy 36→248 px** vì mask seg lem.
- **Chỉ crop khi bàn nhỏ hơn `--roi-min-frac` (35%) chiều cao khung.** Video ngang mẫu
  bàn chiếm 42% → không crop (đã chứng minh crop cho FEN y hệt, crop chỉ thêm rủi ro).
  Video dọc bàn chỉ chiếm 31% → crop có lợi vì letterbox về imgsz 960 làm quân bé lại.
- Seg trả `None` cả loạt → dùng full-frame.
- `--no-roi` để tắt hẳn.

---

## 6. Ngưỡng và lý do

| Cờ | Mặc định | Vì sao |
|---|---|---|
| `--step` | 20 s | đo: 181 frame/28 s decode, thấy đủ mọi mốc ván |
| `--fine-step` | 1 s | đủ để chốt đúng giây trong pha xếp bàn |
| `--fen-step` | 5 s | đánh đổi: 1 s cho chuỗi nước đi tốt hơn nhưng lâu gấp 5 |
| `--conf` | 0.25 | bằng `PIECE_CONFIDENCE_THRESHOLD`, đúng ngưỡng prod |
| `--cand-dist` | **14** | mẫu thô tại mốc ván đo được `d` = 0, 3, 4, 6, **9**; giữa ván 31-40. Đặt 8 là **hụt đúng ván có d=9** |
| `--cand-pieces` | 28 | giữa ván đo được 13-24 quân → 28 không thể chạm tới |
| `--start-dist` | 2 | ở 1 fps, mốc thật cho `d` = 0 |
| `--start-pieces` | 30 | đủ 32 trừ hao 2 quân bị tay che |
| `--cluster-gap` | 30 s | pha xếp bàn nhiễu kéo dài ~15-20 s |
| `--min-gap` | 120 s | ván ngắn nhất trong video mẫu ~3,7 phút |
| `--confirm-*` | 60 s / d≥25 / ≤25 quân / 3 mẫu | chặn thế giữa ván tình cờ giống khai cuộc |
| `--reset-lead` | 25 s | ván trước coi như hết trước khi bắt đầu xếp lại |
| `--lead` / `--tail` | **300 s** | lấy dư 5 phút mỗi đầu cho chắc (yêu cầu của người dùng) |

---

## 7. Kết quả trên video mẫu

`E:\videos\GiangHo\Nhâm Thầy Cúng vs Nghĩa Sing.mp4` — 60,4 phút, 1280x720.

```
decode thô 181 frame / 27 s   →  detect 16 s
bàn chiếm 42% chiều cao -> KHÔNG crop
6 cửa sổ quét tinh -> 1293 mẫu

=== 6 ván ===
  ván 1  00:00:09 -> 00:10:58  (10.8 phút)
  ván 2  00:11:23 -> 00:18:52  ( 7.5 phút)
  ván 3  00:19:17 -> 00:23:00  ( 3.7 phút)
  ván 4  00:23:25 -> 00:42:43  (19.3 phút)
  ván 5  00:43:08 -> 00:52:37  ( 9.5 phút)
  ván 6  00:53:02 -> 01:00:27  ( 7.4 phút)
```

**Đã kiểm chứng bằng mắt**: cả 6 tấm trong `starts.jpg` đều là bàn đủ 32 quân ở thế
khai cuộc. Đồng hồ trong khung hình có ô đếm ván, tới ván cuối hiện `4 | 2` = 6 ván —
khớp đúng số ván tìm được.

Cắt 6 clip mất ~5-18 giây, tổng 1,3 GB (do lấy dư 5 phút mỗi đầu). Mốc thật sau khi
snap keyframe lệch 0-5,4 s so với mốc yêu cầu, **luôn lệch về phía trước**.

Toàn bộ nằm ở `E:\videos\GiangHo\Nhâm Thầy Cúng vs Nghĩa Sing\`, video gốc đã được
dời vào cùng chỗ thành `00_goc_Nhâm Thầy Cúng vs Nghĩa Sing.mp4`.

---

## 8. Giới hạn đã biết

| Vấn đề | Hiện trạng |
|---|---|
| **Video bình luận có cắt cảnh / zoom / bàn 2D overlay** | Chưa đo. ROI cố định sẽ sai khi cắt cảnh → dùng `--no-roi`, và luôn duyệt `starts.jpg` trước khi cắt |
| **Bàn phân tích 2D lặp lại thế khai cuộc** | Sinh mốc giả. Chặn bằng `--min-gap` + xác nhận "ván trước đã tàn"; ứng viên bị loại ghi ở `rejected.json` |
| **Livestream có đoạn dài không có bàn** (chầu, nói chuyện) | Đoạn đó vẫn nằm trong clip của ván trước. Chưa cắt tự động; nếu vướng thì thêm luật "mất bàn quá N giây thì cắt đuôi" |
| **Ván đấu lại ngay trong `--min-gap`** | Bị gộp. Hạ `--min-gap` nếu video có nhiều ván siêu ngắn |
| **Bàn bị đọc ngược đầu (đổi màu)** | Lệch đúng 32 ô nên nằm lẫn trong dải "không khai cuộc". Script đo riêng khoảng cách tới thế khai cuộc **đổi màu** và in cảnh báo nếu gặp |
| **Chuỗi FEN chưa phải nước đi** | Xem §4 |

## 9. Liên quan

- `scripts/_fenutil.py` — helper FEN dùng chung (`expand_rows`, `mirror_fen`,
  `cell_diff`, `dist_to_start`, `dist_to_start_farside`).
- `boarddetection/pipeline.py` — `recognize_image_2pass`, `passes_gate`.
- `docs/ROT180_TRAINING_GAP.md` — vì sao có lượt đọc thứ hai (bản xoay 180°).
