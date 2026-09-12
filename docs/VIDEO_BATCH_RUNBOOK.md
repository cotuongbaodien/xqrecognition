# Runbook — cắt kho video thành clip từng ván

Cập nhật **2026-09-12** (tối). Đây là tài liệu vận hành: đang ở đâu, chạy tiếp thế nào,
và những cái bẫy đã trả giá. Công cụ chi tiết xem [`VIDEO_SPLIT.md`](VIDEO_SPLIT.md),
danh sách nguồn xem [`VIDEO_SOURCES.md`](VIDEO_SOURCES.md).

---

## 1. Đang ở đâu (2026-09-12, tối)

| | |
|---|---|
| **Video đã cắt xong** | **768** |
| **TỔNG SỐ VÁN** | **3 996** |
| Giờ nội dung đã xử lý | 1 113,3 giờ |
| Dung lượng clip | 818 GB |
| Video còn chờ cắt | **0** |
| Tên thư mục | **100% đã có tiền tố `YYYYMMDD_`** (§11) |
| Chỗ trống | E: 590 GB · D: 755 GB |

Không còn tiến trình nào đang chạy. Việc tiếp theo: §4.

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

> **Đừng dời file đang copy dở.** Người dùng hay gửi qua SendAnywhere: file nửa chừng có
> đuôi `.sendanywhere` (bộ lọc đuôi video tự loại), nhưng file `.MOV` vừa xuất hiện vẫn
> có thể đang được ghi tiếp. Chụp kích thước, chờ ~20 s, chụp lại — **chỉ dời file không
> đổi kích thước**, và thử `open(p,"rb")` để chắc Windows không khoá.

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

1. **Nén clip** — `video_shrink.py`, thu hồi ~500 GB trong 818 GB. Chạy được ngay,
   không đụng bản gốc, an toàn cả khi đang cắt (đòi `index.csv` + `--min-age`).
2. **Quét dày cho ~354 ván vẫn dài > 30 phút** — `video_resplit_long.py --gap-step 4`
   (~6 phút/thư mục). Đây là bước đắt, chỉ chạy cho chỗ còn sót (§9).
3. **24 thư mục bị cổng chặn loại** khi tách lại — soi tay, xem mốc nào bị mất và vì sao.
4. **Lọc ván trùng bằng chuỗi FEN** trước khi đăng — §7, bắt buộc.
5. Kho TikTok (774 giờ) — đợt cuối, đúng thứ tự ưu tiên giang hồ → YouTube → TikTok.

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
| `UnicodeEncodeError: 'gbk' codec` | chạy qua shell không phải UTF-8, log tiếng Việt chết ngay dòng đầu | đặt `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` khi chạy nền |
| Nới ngưỡng lại MẤT ván | cụm dài ra → mốc chốt trôi → dính luật gộp | cổng chặn mất mốc (§9), mặc định bật |

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

## 9. Ván dài bất thường — đã đo trên cả kho

**415 ván dài hơn 30 phút** nằm ở 321 thư mục. Ván cờ chớp không thể dài vậy: lượt cắt
đã **bỏ sót mốc giữa chừng** (lúc xếp lại bàn tay che kín, hoặc hai bên đi ngay khi vừa
xếp xong nên không mẫu 20 s nào rơi trúng thế khai cuộc).

Chữa theo **hai bước, bước rẻ trước**. **Bước 1 đã chạy xong 12/09** (134 thư mục,
+254 ván); bước 2 còn ~354 ván chưa làm.

```bat
::  BƯỚC 1 — chỉ nới ngưỡng, tính lại trên cache sẵn có. Không nạp model, chưa tới 1 phút.
python scripts\video_resplit_long.py --no-rescan                  :: xem trước
python scripts\video_resplit_long.py --no-rescan --apply          :: cắt lại

::  BƯỚC 2 — quét dày 4 s/frame đúng khoảng còn dài (~6 phút/thư mục, chỉ cho chỗ còn sót)
python scripts\video_resplit_long.py --min-long 30 --gap-step 4
```

Kết quả đo bước 1 trên 322 thư mục:

| | |
|---|---|
| Tách thêm được, **không mất mốc cũ nào** — ĐÃ CẮT LẠI | **134 thư mục → +254 ván** |
| Bị cổng chặn loại (xem dưới) | 24 thư mục |
| Không đổi | 165 thư mục |
| Ván > 30 phút trong nhóm tách được | 191 → 130 |

Cắt lại 134 thư mục mất ~45 phút (ffmpeg copy), **0 lỗi**. Kho đi từ 3 714 lên 3 996 ván
(gồm cả 13 video mới trong ngày). Dung lượng clip 621 → 818 GB vì mỗi clip vẫn lấy dư
2 phút hai đầu — `video_shrink.py` thu hồi lại được cỡ 500 GB.

**Ngưỡng lỏng hơn KHÔNG bảo đảm ra nhiều mốc hơn.** Thêm mẫu vào một cụm làm cụm dài
ra, mốc chốt (mẫu cuối cụm) trôi về sau rồi dính luật gộp `--min-gap` và biến mất —
**24/322 thư mục mất tổng 28 mốc đang có**. Vì vậy công cụ mặc định **bỏ qua nguyên thư
mục nào làm mất mốc cũ**; `--cho-mat-moc` để tắt, `--giu-tol` đổi dung sai (mặc định 60 s).

Độ tin cậy 253 mốc mới (lấy mẫu cache gần nhất): **245/252 lệch ≤ 6 ô** so với thế khai
cuộc, **236/252 đủ ≥ 30 quân** — đúng chữ ký bàn vừa xếp lại. Đã duyệt mắt 36 mốc bằng
ảnh ghép: 30 mẫu ngẫu nhiên đều là bàn khai cuộc sạch; 6 ca "yếu" là frame rơi trúng lúc
tay còn đang xếp hoặc màn hình splash của app — vẫn là ranh giới ván thật.

`--apply` xoá clip cũ + `index.csv` của đúng thư mục có thay đổi rồi cắt lại; bản gốc
`00_goc_*` **không bao giờ bị đụng**, nên sai vẫn lùi được.

## 10. Khi nào cần ổ D

Chỉ khi E: xuống dưới ~200 GB. `yt_download.py` **tự chuyển sang D:** khi ổ chính còn
dưới `--min-free` (mặc định 150 GB). Với kho clip thì dời bớt thư mục đã xong sang D:

```bat
move "E:\videos\GiangHo\<thư mục đã xong>" "D:\videos\GiangHo_done\"
```

**Chưa xoá bản gốc nào** — mọi video gốc vẫn nằm trong thư mục của nó dưới tên
`00_goc_*`, để lỡ mốc ván sai còn cắt lại.

## 11. Tên thư mục — quy ước `YYYYMMDD_tên`

Thư mục tải từ YouTube vốn đã có dạng `20260423_tên [id]`; thư mục từ file quay tay thì
không, trộn vào là hết sắp theo thời gian. Đã chuẩn hoá **248 thư mục**, giờ cả 768 đều
có tiền tố ngày.

```bat
python scripts\video_rename_date.py --dry-run     :: xem sẽ đổi thành gì
python scripts\video_rename_date.py               :: đổi thật
```

Ngày lấy theo độ tin cậy giảm dần, vì **mỗi nguồn sai một kiểu**:

| Nguồn | Số thư mục | Vì sao xếp ở đó |
|---|---|---|
| `com.apple.quicktime.creationdate` | 97 | iPhone ghi đúng lúc bấm quay |
| `creation_time` trong container | 95 | tin được, nhưng file đã re-encode thì là ngày encode |
| Cụm 6 số `YYMMDD` trong tên | 6 | quy ước của kho; đối chiếu được: `minhla26042302` nằm trong thư mục YouTube `20260423_...` |
| Thừa hưởng từ bản cùng nhóm | 11 | `... - P1` chỉ còn mtime thì lấy ngày của bản đầy đủ |
| `mtime` của file gốc | 38 | **chốt cuối** — đây là lúc copy/chia file, không phải lúc quay |

Hai chỗ ngày trong TÊN lệch với metadata, đã theo metadata: `caotienminhla260908` →
**20260818**, `minhla250426` → **20260424**. Cách đặt tên trong kho không nhất quán
(vừa `YYMMDD` vừa `DDMMYY`) nên tên chỉ dùng để tham khảo, không dùng làm nguồn chính.

Chỉ đổi tên **thư mục**, tên clip bên trong giữ nguyên: `index.csv` lưu tên file nên
không vỡ, `_data/scan.json` có đường dẫn tuyệt đối nhưng đã cũ từ trước (bản gốc đã
được dời vào trong thư mục thành `00_goc_*`, mọi công cụ tìm qua `src_of()`).

⚠ Thư mục nào còn **file nguồn cùng tên ở tầng ngoài** thì không được đổi — `video_split.py`
so tên file với tên thư mục để biết đã cắt chưa, đổi tên là nó cắt lại từ đầu. Công cụ
tự bỏ qua những thư mục đó.
