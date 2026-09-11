# Runbook — cắt kho video thành clip từng ván

Cập nhật **2026-09-12**. Đây là tài liệu vận hành: đang ở đâu, chạy tiếp thế nào,
và những cái bẫy đã trả giá. Công cụ chi tiết xem [`VIDEO_SPLIT.md`](VIDEO_SPLIT.md),
danh sách nguồn xem [`VIDEO_SOURCES.md`](VIDEO_SOURCES.md).

---

## 1. Đang ở đâu (2026-09-12)

| | |
|---|---|
| **Video đã cắt xong** | **481** |
| **TỔNG SỐ VÁN** | **1948** |
| Giờ nội dung đã xử lý | 513,8 giờ |
| Dung lượng clip | 597 GB |
| Video chờ cắt (tầng ngoài `E:\videos\GiangHo`) | ~168 |
| YouTube đã tải | 461 / 580 |
| File tải xong chờ gộp (`E:\videos\_yt_tai`) | 66 · 22 GB |
| Chỗ trống | E: 937 GB · D: 812 GB |

Số liệu sống: mở `E:\videos\GiangHo\_BAO_CAO.md`, hoặc chạy
`python scripts/video_report.py`.

## 2. Video mới — quy trình cố định

Người dùng bỏ video mới vào **`E:\videos\newvideo`** rồi báo. Việc cần làm:

```python
# dời vào kho, đổi tên nếu trùng (trùng tên FILE hoặc trùng tên THƯ MỤC kết quả)
import os, shutil
src, dst = r"E:\videos\newvideo", r"E:\videos\GiangHo"
for f in sorted(os.listdir(src)):
    p = os.path.join(src, f)
    if not os.path.isfile(p) or not f.lower().endswith(
            (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".ts", ".webm")):
        continue
    stem, ext = os.path.splitext(f); target, k = os.path.join(dst, f), 2
    while os.path.exists(target) or os.path.isdir(os.path.join(dst, stem)):
        stem = f"{os.path.splitext(f)[0]}_{k}"; target = os.path.join(dst, stem + ext); k += 1
    shutil.move(p, target)
```

Rồi chạy lệnh cắt ở §3. Video đã cắt tự bị bỏ qua nên cứ chạy cả thư mục.

> Tiến trình cắt **đọc danh sách MỘT LẦN lúc khởi động**. Video dời vào sau đó phải
> đợi lượt chạy kế tiếp — không tự nhận giữa chừng.

## 3. Lệnh chạy

```bat
cd C:\Resources\xqrecognition

::  CẮT — chỉ chạy MỘT tiến trình, xem §5 trước
python scripts\video_split.py "E:\videos\GiangHo" --fen-step 0 --lead 120 --tail 120 --reencode never

::  TẢI YouTube phần còn thiếu (tự bỏ qua cái đã tải)
python scripts\yt_download.py output\yt_can_tai.txt

::  SOÁT clip (bắt clip ghi dở)
python scripts\video_verify.py "E:\videos\GiangHo"

::  NÉN clip cho nhẹ (~1/3 dung lượng)
python scripts\video_shrink.py "E:\videos\GiangHo"

::  BÁO CÁO số ván
python scripts\video_report.py

::  DÒ FILE TRÙNG TUYỆT ĐỐI giữa các nguồn
python scripts\video_dupes.py
```

Gộp file tải về vào kho rồi cắt: dùng đoạn code ở §2 nhưng `src = E:\videos\_yt_tai`
(và `D:\videos\_yt_tai` nếu có).

**Dừng sạch giữa chừng**: `echo. > E:\videos\GiangHo\_STOP` — cắt xong video đang làm
rồi thoát. Đừng giết ngang.

## 4. Thứ tự việc còn lại

1. Cắt nốt ~168 video đang chờ (gồm 10 video mới ngày 12/09).
2. Gộp 66 file YouTube đã tải → cắt tiếp.
3. Tải nốt 119 video YouTube còn thiếu → gộp → cắt.
4. Soát (`video_verify`) rồi nén (`video_shrink`) — thu hồi ~400 GB.
5. **Lọc ván trùng bằng chuỗi FEN** trước khi đăng — xem §7, đây là việc bắt buộc.
6. Kho TikTok (774 giờ) — đợt cuối.

## 5. BA LUẬT SỐNG CÒN (đều đã trả giá)

**① Chỉ MỘT tiến trình cắt tại một thời điểm.** Kiểm trước khi chạy:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*video_split.py*' -and $_.Name -eq 'python.exe' } | Select-Object ProcessId
```

Hai tiến trình cùng thư mục sẽ xoá thư mục frame tạm của nhau (`Could not open file
...jpg`) và khoá file của nhau (`WinError 32`).

**② Dừng tiến trình nền KHÔNG giết vòng giám sát.** Đã có lúc tồn tại **7 vòng
`run_all.sh` + 2 tiến trình cắt** cùng lúc vì mỗi lần khởi động lại, vòng cũ vẫn sống
và tiếp tục đẻ tiến trình mới. Phải diệt theo PID:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'bash.exe' -and $_.CommandLine -match 'run_all|wait_ram' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

⚠ Lọc phải kèm **tên tiến trình**. Lọc chỉ theo CommandLine thì chính câu lệnh
PowerShell/shell đang chạy cũng khớp → tự giết mình, hoặc chờ mãi một tiến trình
không tồn tại (pipeline từng đứng 6 tiếng vì lỗi này).

**③ Cần ít nhất ~6 GB RAM trống.** Máy 31 GB nhưng qemu + WSL + java + chrome ăn hết,
có lúc chỉ còn 3,6 GB → Windows giết tiến trình cắt **4 lần liên tiếp**. Dấu hiệu nhận
biết: hàng loạt `ffprobe lỗi:` **với thông báo TRỐNG TRƠN** (không spawn nổi tiến
trình, không phải file hỏng). Muốn nhẹ RAM thì đặt `OCR_MODEL_FORMAT=onnx` (chậm hơn
~25%, không dùng GPU).

## 6. Sự cố đã gặp và cách chặn

| Sự cố | Nguyên nhân | Đã chặn bằng |
|---|---|---|
| Mốc ván lệch 3-6 phút | nhiều lượt decode ghi chung thư mục frame | thư mục riêng theo tag **+ PID** |
| `Could not open file …jpg` | hai tiến trình cắt xoá frame của nhau | PID trong tên thư mục |
| `WinError 32` khoá file | hai tiến trình cắt cùng video | luật ① |
| `CUDA illegal memory access` | job transcribe của người dùng chiếm hết VRAM | dừng job kia, hoặc `OCR_MODEL_FORMAT=onnx` |
| `Expecting ',' delimiter` | cache bị giết đúng lúc đang ghi | ghi nguyên tử (`.tmp` + đổi tên) |
| `ffprobe lỗi:` trống trơn ×130 | hết RAM, không spawn nổi tiến trình | `_run()` thử lại 3 lần khi mã≠0 **và** stderr rỗng |
| Mất một ván khi "tối ưu" decode | nới ngưỡng keyframe-only quá tay | ngưỡng `keyframe ≤ step/8` |
| Nén AV1 chỉ nhỏ đi 2% | encode AV1 → H.264 là đổi xuôi thành ngược | bỏ qua mọi file bitrate thấp |
| Báo "0 video tải được" | đếm theo mã thoát, mà yt-dlp thoát ≠0 khi có bất kỳ video nào lỗi | đếm chênh lệch dòng trong `_archive.txt` |
| Vòng báo cáo tự chết | pipeline và vòng watch cùng ghi một file | ghi nguyên tử |

## 7. Ván trùng — PHẢI xử lý trước khi đăng

Đã đo, không phải phỏng đoán:

- **5 buổi có CẢ bản đầy đủ LẪN các phần** trong kho (`caotienminhla260818`,
  `minhla250426`, `minhlacaotien`, `minhlaphucloi`, `MinhLaSonHang Manh`) →
  **~99 ván bị cắt hai lần**. Xác minh: `MinhLaSonHang Manh - P1` tại giây 60 khớp
  **lệch 0 ô** với bản đầy đủ tại giây 60.
- **19 file nguồn trùng tuyệt đối** (26,3 GB) — `video_dupes.py` đã dò ra, chưa xoá.
- Ba nguồn chồng nhau (cục bộ / YouTube / TikTok) nên còn trùng nữa sau khi cắt.

**Tên file và dung lượng KHÔNG bắt được trùng ở mức ván** — hai clip cùng một ván từ
hai nguồn có tên khác hẳn, độ dài khác, dung lượng khác. Phải so bằng **chuỗi FEN**:
lấy vài chục thế cờ đầu mỗi clip làm vân tay, ván nào trùng thì gom nhóm, giữ bản nét
nhất. Chạy `video_split.py --stage segment --fen-step 2` cho từng video để có chuỗi FEN.

## 8. File chia phần — đã kiểm, gần như không mất ván

Buổi live 8-9 tiếng hay bị chia thành `_p1.._p8`. Lo ngại: chỗ chia rơi vào giữa ván.

Đo bằng thế cờ trên 31 ranh giới (`video_stitch_parts.py --dry-run`):
**30/31 chỗ lệch 17-42 ô = hai ván khác nhau**, chỉ **1 chỗ** thật sự đứt một ván.
Các phần cũng **không nối tiếp nhau** (P2 không bắt đầu ở chỗ P1 kết thúc), nên chúng
là những đoạn trích rời chứ không phải một lát cắt tuần tự.

Muốn khâu chỗ đứt đó: `python scripts/video_stitch_parts.py` (giữ nguyên hai nửa gốc).

## 9. Ván dài bất thường

~25 ván dài hơn 30 phút — nhiều khả năng **hai ván bị dính làm một** do bỏ sót mốc.
Tính lại mốc không cần cắt lại (~1 giây/video):

```bat
python scripts\video_split.py "E:\videos\GiangHo\<thư mục>" --stage segment --start-dist 4 --gap-step 4
```

Xem `starts.jpg` thấy ổn mới `--stage cut --overwrite`.

## 10. Khi nào cần ổ D

Chỉ khi E: xuống dưới ~200 GB. `yt_download.py` **tự chuyển sang D:** khi ổ chính còn
dưới `--min-free` (mặc định 150 GB). Với kho clip thì dời bớt thư mục đã xong sang D:

```bat
move "E:\videos\GiangHo\<thư mục đã xong>" "D:\videos\GiangHo_done\"
```

**Chưa xoá bản gốc nào** — mọi video gốc vẫn nằm trong thư mục của nó dưới tên
`00_goc_*`, để lỡ mốc ván sai còn cắt lại.
