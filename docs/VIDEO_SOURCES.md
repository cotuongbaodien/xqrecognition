# Danh sách nguồn video + thứ tự làm

Cập nhật 2026-09-11. Kèm với [`VIDEO_BATCH_RUNBOOK.md`](VIDEO_BATCH_RUNBOOK.md)
(cách chạy) và [`VIDEO_SPLIT.md`](VIDEO_SPLIT.md) (công cụ hoạt động ra sao).

---

## 1. Toàn bộ nguồn — ĐÃ ĐO ĐỘ DÀI THẬT

| # | Nguồn | Video | Giờ | Trạng thái |
|---|---|---|---|---|
| 1 | `E:\videos\GiangHo` | 228 | **415,9** | **đang cắt** — 93 xong (342 ván), 136 còn lại |
| 2 | `E:\Tiktoker\cotuongnghiasing\tiktoklive\raw` | 159 | **452,3** | chờ (đợt 3) |
| 3 | `E:\Tiktoker\cotuongnghiasing\download` | 112 | **286,2** | chờ (đợt 3) |
| 4 | `E:\Tiktoker\cotuongnghiasing\gianghorecord` | 15 | **35,9** | chờ (đợt 3) |
| 5 | `E:\Tiktoker\cotuongnghiasing\tiktoklive\edit` + `post` | 88 | **29,8** | cân nhắc bỏ — nghi là đoạn trích của (2) |
| | **CỘNG CỤC BỘ** | **602** | **1220,1** | |
| 6 | YouTube `@Cotuongnghiasing` | 1026 | **1575,0** | 446 video (868 giờ) **đã có ở máy** |
| | **CỘNG KHÔNG TRÙNG** | **1182** | **~1926** | = 1220 cục bộ + 706 YouTube chưa có |

Cộng thô cả hai là 2795 giờ, nhưng **đếm trùng**: kênh YouTube là bản backup của chính
kho cục bộ. Số thật là **~1926 giờ ≈ 80 ngày phát liên tục**.

## 2. YouTube — 713 video ẨN, phải đi đúng đường mới thấy

| Đường | Số video |
|---|---|
| tab `/videos` + `/streams` (kể cả có cookie) | 314 |
| **uploads playlist `UUs5-T_IsPmBEBwiGuzyw-vw`** | **1028** |

Video ẩn (unlisted) **không hiện trên tab kênh dù đã đăng nhập** — chỉ hiện trong
uploads playlist. Lệnh đúng:

```bash
yt-dlp --cookies <file> --extractor-args "youtubetab:skip=authcheck" \
       "https://www.youtube.com/playlist?list=UUs5-T_IsPmBEBwiGuzyw-vw"
```

### Cookie — hai cái bẫy đã dính

1. **File cookie Netscape bắt buộc ngăn cách bằng TAB.** Paste qua chat thì tab
   thành dấu cách, yt-dlp bỏ sạch mọi dòng (`invalid length 1`) và **chạy như chưa
   đăng nhập** mà không báo lỗi rõ ràng — chỉ thấy đủ 314 video.
2. **yt-dlp GHI ĐÈ file cookie** sau mỗi lần chạy. Giữ bản gốc riêng, mỗi lần chạy
   copy ra bản làm việc:
   - gốc: `C:\Users\PC\.yt_cookies_master.txt`
   - làm việc: `C:\Users\PC\.yt_cookies_work.txt`

Cookie là **khoá phiên đăng nhập Google** — để ngoài repo, không commit, dùng xong
nên xoá hoặc đăng xuất phiên đó.

## 3. Chỗ chứa — đã bớt căng sau khi lọc trùng

Đối chiếu độ dài (±3 giây) giữa 1026 video YouTube và 602 file cục bộ:

| | Video | Giờ | Tải 720p |
|---|---|---|---|
| YouTube đã có ở máy | 446 | 868 | — |
| **YouTube CHƯA có** | **580** | **706** | **~477 GB** |

Chỉ cần tải **477 GB** chứ không phải ~1060 GB như ước ban đầu. Phân bố để chia đợt:

| Độ dài | Video | Giờ | Tải 720p |
|---|---|---|---|
| dưới 10 phút | 92 | 5 | ~4 GB |
| 10-30 phút | 108 | 35 | ~23 GB |
| 30-60 phút | 131 | 101 | ~68 GB |
| 1-2 giờ | 120 | 170 | ~115 GB |
| trên 2 giờ | 129 | 395 | ~267 GB |

Danh sách id: `output/yt_can_tai.txt` (đã sắp NGẮN TRƯỚC, `yt-dlp -a` nhận thẳng).

> ⚠ 706 giờ là **cận trên**. File cục bộ nhiều cái là `part1..5` cắt từ một buổi live
> dài, còn YouTube giữ nguyên buổi — những cặp đó không khớp độ dài dù cùng nội dung,
> nên bị tính nhầm là "chưa có". Tải xong nên soi lại vài cái trước khi tin hết.

| Chỗ trống | |
|---|---|
| E: | ~1547 GB |
| D: | 811 GB |
| Cắt nốt (1) cần | ~700 GB |
| Tải YouTube phần thiếu | ~477 GB, cắt ra thêm ~620 GB |

## 4. Thứ tự làm — ưu tiên GIANG HỒ → YOUTUBE → TIKTOK

Thứ tự do người dùng chốt 2026-09-11.

### Đợt 1 — Giang Hồ (đang chạy)
1. **Cắt nốt `E:\videos\GiangHo`** — 136 video còn lại, ~12-15 tiếng.
2. **Soát** (`video_verify.py`) rồi **nén** (`video_shrink.py`) — thu hồi ~430 GB.
   Phải xong bước nén trước khi kéo nguồn mới về, vì nó trả lại chỗ.
3. **Soi 22 ván dài hơn 30 phút** (nghi dính hai ván làm một): chạy lại
   `--stage segment --start-dist 4 --gap-step 4`, xem `starts.jpg` rồi mới cắt lại.

### Đợt 2 — YouTube (1028 video, 1574,7 giờ)
4. **Lọc trùng trước khi tải** — `scripts/video_dedup_yt.py`. Kênh là bản backup của
   chính kho cục bộ nên phần lớn nhiều khả năng đã có; chỉ tải phần thiếu.
5. **Tải theo đợt ~150-200 video**, ngắn trước dài sau, mỗi đợt cắt + nén xong mới
   sang đợt kế — để không bao giờ giữ đồng thời cả bản tải lẫn clip chưa nén.
6. Tải xong đợt nào thì **dời file vào `E:\videos\GiangHo`** rồi chạy lệnh cắt như cũ.

### Đợt 3 — TikTok / kho cục bộ còn lại
7. **Gộp `download` (112) + `tiktoklive
aw` (159) + `gianghorecord` (15)** vào
   `E:\videos\GiangHo` rồi cắt. Đã kiểm: **2 file trùng tên** với kho hiện có →
   đổi tên khi dời, đừng ghi đè.
8. Quyết sau: có lấy `tiktoklive\edit` (87) + `post` (1) không — nghi là đoạn trích
   của `raw`, cắt lại sẽ ra clip trùng.

### Sau cùng
9. **Chuỗi FEN** cho video nào cần biên bản ván cờ — `--fen-step 2`.

## 5. Lệnh gộp nguồn cục bộ (bước 3)

```python
# đổi tên nếu trùng, dời trong cùng ổ nên tức thì
import os, shutil
dst = r"E:\videos\GiangHo"
for src in [r"E:\Tiktoker\cotuongnghiasing\download",
            r"E:\Tiktoker\cotuongnghiasing\tiktoklive\raw",
            r"E:\Tiktoker\cotuongnghiasing\gianghorecord\records",
            r"E:\Tiktoker\cotuongnghiasing\gianghorecord\pending"]:
    for f in os.listdir(src):
        if not f.lower().endswith((".mp4",".mov",".mkv",".avi",".m4v",".ts")):
            continue
        target = os.path.join(dst, f)
        stem, ext = os.path.splitext(f)
        k = 2
        while os.path.exists(target) or os.path.isdir(os.path.join(dst, stem)):
            target = os.path.join(dst, f"{stem}_{k}{ext}"); stem = f"{stem}_{k}"; k += 1
        shutil.move(os.path.join(src, f), target)
```

Sau đó chạy lại đúng lệnh cắt ở runbook — video đã xử lý tự bị bỏ qua.

## 6. Chưa quyết

- Có lấy `tiktoklive\edit` + `post` không (nghi là đoạn trích của `raw`).
- Chất lượng tải YouTube: 720p hay 480p.
- Có xoá bản tải về sau khi cắt không.
- Bản gốc cục bộ: **giữ nguyên**, chưa xoá gì cho tới khi soi đủ.
