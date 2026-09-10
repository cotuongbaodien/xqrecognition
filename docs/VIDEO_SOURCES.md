# Danh sách nguồn video + thứ tự làm

Cập nhật 2026-09-11. Kèm với [`VIDEO_BATCH_RUNBOOK.md`](VIDEO_BATCH_RUNBOOK.md)
(cách chạy) và [`VIDEO_SPLIT.md`](VIDEO_SPLIT.md) (công cụ hoạt động ra sao).

---

## 1. Toàn bộ nguồn

| # | Nguồn | Số video | Dung lượng / độ dài | Trạng thái |
|---|---|---|---|---|
| 1 | `E:\videos\GiangHo` | 229 | 416,9 giờ · 656 GB | **đang cắt** — 93 xong (342 ván), 136 còn lại |
| 2 | `E:\Tiktoker\cotuongnghiasing\download` | 112 | 149,6 GB | chờ — gộp vào (1) rồi cắt |
| 3 | `E:\Tiktoker\cotuongnghiasing\tiktoklive\raw` | 159 | 225,1 GB | chờ |
| 4 | `E:\Tiktoker\cotuongnghiasing\gianghorecord\records` | 14 | 22,4 GB | chờ |
| 5 | `E:\Tiktoker\cotuongnghiasing\gianghorecord\pending` | 1 | 0,4 GB | chờ |
| 6 | `E:\Tiktoker\cotuongnghiasing\tiktoklive\edit` | 87 | 36,8 GB | **cân nhắc bỏ** — bản đã dựng/cắt sẵn, nhiều khả năng là đoạn trích của (3) |
| 7 | `E:\Tiktoker\cotuongnghiasing\tiktoklive\post` | 1 | 1,7 GB | như trên |
| 8 | YouTube `@Cotuongnghiasing` | **1028** | **1574,7 giờ** | chờ tải — xem §2 |
| | **Cộng file cục bộ (1-5)** | **515** | **~1054 GB** | |

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

## 3. Chỗ chứa — đây mới là ràng buộc

| | |
|---|---|
| E: trống | ~1547 GB |
| D: trống | 811 GB |
| (1) cắt nốt cần | ~700 GB |
| (2)-(5) cắt cần | ~1370 GB clip |
| (8) tải 720p (~1,5 Mbps) | **~1060 GB**, cắt ra thêm ~1380 GB |

⇒ **Không đủ chỗ để làm tất cả cùng lúc.** Bốn hướng, chọn trước khi tải:

- **Lọc trùng trước.** Kênh YouTube là bản backup của chính đám file cục bộ, nên
  phần lớn 1028 video nhiều khả năng đã có ở (1)-(5). Đối chiếu theo **độ dài ±3 giây**
  rồi chỉ tải phần thiếu — hướng này rẻ nhất, làm trước khi tải bất cứ thứ gì.
- **Tải 480p** (~0,8 Mbps → ~570 GB): detector đọc bàn cờ ở 480p vẫn ổn với bàn chiếm
  hơn 1/3 khung, nhưng **chưa đo** — phải thử vài video trước.
- **Cắt xong thì xoá bản tải về**, chỉ giữ clip (kênh YouTube vẫn là bản gốc trên mây).
- **Chia đợt**: tải + cắt 200 video một đợt, nén, rồi mới tới đợt sau.

## 4. Thứ tự làm

1. **Cắt nốt (1)** — đang chạy, ~12-15 tiếng. Không đụng gì thêm cho tới khi xong.
2. **Nén clip đã cắt** — `video_shrink.py`, thu hồi ~430 GB. **Phải xong bước này
   trước khi thêm nguồn mới**, vì nó trả lại chỗ.
3. **Gộp (2)-(5) vào `E:\videos\GiangHo`** rồi chạy lại lệnh cắt. Lưu ý:
   - Đã kiểm: **2 file trùng tên** với (1) → đổi tên khi dời, đừng ghi đè.
   - Dời trong cùng ổ E: là đổi tên, tức thì.
4. **Soát + nén** đợt đó.
5. **Lọc trùng YouTube** (§3) → chốt danh sách thật sự cần tải.
6. **Tải + cắt YouTube theo đợt**, mỗi đợt nén xong mới sang đợt kế.
7. **Soi 22 ván dài hơn 30 phút** (nghi dính hai ván làm một) — `--stage segment`
   với `--start-dist 4 --gap-step 4`, xem `starts.jpg` rồi mới cắt lại.
8. **Chuỗi FEN** cho video nào cần biên bản — `--fen-step 2`.

Bước 6 chỉ bắt đầu khi bước 2 và 4 đã trả đủ chỗ trống.

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
