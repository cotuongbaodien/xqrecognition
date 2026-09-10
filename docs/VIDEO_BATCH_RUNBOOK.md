# Runbook — chạy lô cắt video kho `E:\videos\GiangHo`

Bản hướng dẫn để **tự chạy tiếp** không cần hỏi lại. Cập nhật 2026-09-10.
Chi tiết công cụ: [`VIDEO_SPLIT.md`](VIDEO_SPLIT.md).

---

## 1. Đang ở đâu

| | |
|---|---|
| Kho | `E:\videos\GiangHo` — **229 video, 416,9 giờ, 656 GB** |
| Đã cắt | **~78 video · 56,6 giờ · ~260 ván** (~120 GB clip) |
| Còn lại | **~150 video · ~360 giờ · 550 GB nguồn** |
| Nén | mới nén ~30 clip rồi tạm dừng để nhường GPU cho việc cắt |
| Chỗ trống | E: ~1579 GB · D: ~811 GB |

Clip copy-stream của phần còn lại sẽ chiếm **~715 GB** → cắt xong E: còn ~860 GB.
**Chưa cần dùng ổ D.**

## 2. Chạy tiếp (việc chính)

Mở terminal ở `C:\Resources\xqrecognition`:

```bat
python scripts\video_split.py "E:\videos\GiangHo" --fen-step 0 --lead 120 --tail 120 --reencode never
```

- Tự **bỏ qua video đã cắt** (thư mục có `index.csv`) nên chạy lại lúc nào cũng được.
- Ngắt giữa chừng thoải mái: video dở sẽ làm lại từ khúc quét còn dang dở.
- `--reencode never` = **copy stream, nhanh nhất** (~3 phút xử lý cho mỗi giờ video).
  Ước **15-20 tiếng** cho 360 giờ còn lại.

Muốn nó tự chạy lại khi tiến trình chết (lỗi CUDA lâu lâu vẫn xảy ra):

```bat
for /L %i in (1,1,40) do python scripts\video_split.py "E:\videos\GiangHo" --fen-step 0 --lead 120 --tail 120 --reencode never && goto :done
:done
```

### Dừng SẠCH giữa lô

Tạo file rỗng tên `_STOP` trong `E:\videos\GiangHo`. Nó cắt xong video đang làm rồi
thoát, và tự xoá `_STOP`. Đừng giết ngang tiến trình — dễ để lại clip ghi dở.

```bat
echo. > E:\videos\GiangHo\_STOP
```

## 3. Sau khi cắt xong

**a) Soát lại** (bắt clip ghi dở, thiếu/thừa file):

```bat
python scripts\video_verify.py "E:\videos\GiangHo"
python scripts\video_verify.py "E:\videos\GiangHo" --fix    ::  xoá clip hỏng + index để cắt lại
```

**b) Nén cho nhẹ** (~10-12 tiếng, thu hồi ~430 GB):

```bat
python scripts\video_shrink.py "E:\videos\GiangHo" --dry-run   ::  xem trước
python scripts\video_shrink.py "E:\videos\GiangHo"
```

Chỉ đụng clip `*_vanNN_*`, **không bao giờ đụng `00_goc_*`**. Encode xong nó so lại
độ dài mới dám thay; lệch quá 2 giây là bỏ bản mới, giữ file cũ.

**c) Chuỗi FEN** (làm sau cùng, chỉ khi cần biên bản ván cờ):

```bat
python scripts\video_split.py "<đường dẫn video>" --stage segment --fen-step 2
```

## 4. Ba luật phải nhớ

1. **Chỉ chạy MỘT tiến trình cắt tại một thời điểm.** Hai tiến trình cùng thư mục sẽ
   xoá file frame tạm của nhau, ffmpeg báo `Could not open file ...jpg` rất khó đoán.
   Kiểm tra: `tasklist | findstr python`.
2. **Đừng chạy chung với job dùng GPU khác** (vd `transcribe_one.py`). VRAM đầy thì
   detect chết bằng `CUDA error: an illegal memory access`. Muốn chạy chung thì ép
   phần detect sang CPU: đặt biến môi trường `OCR_MODEL_FORMAT=onnx` (chậm hơn ~25%
   nhưng không đụng GPU).
3. **Nén và cắt đừng chạy song song.** Cả hai ăn cùng khối encoder NVENC, mỗi việc
   chạy nửa tốc. Cắt xong hẵng nén.

## 5. Khi nào cần ổ D

Chỉ khi E: xuống dưới ~200 GB. Cách đơn giản nhất là **dời bớt thư mục đã xong** sang
D rồi cắt tiếp — công cụ chỉ tìm video ở tầng đầu của `E:\videos\GiangHo` nên thư mục
kết quả nằm đâu cũng không ảnh hưởng:

```bat
move "E:\videos\GiangHo\<tên thư mục đã xong>" "D:\videos\GiangHo_done\"
```

**Chưa xoá gì cả** — bản gốc vẫn nằm nguyên trong từng thư mục dưới tên `00_goc_*`,
để lỡ mốc ván sai còn cắt lại. Quyết định xoá bản gốc để dành sau khi soi đủ.

## 6. Kiểm tra nhanh bất cứ lúc nào

```bat
::  đang tới đâu
type E:\videos\GiangHo\_video_split.log | findstr /C:"xong sau"

::  đếm ván
python -c "import os;r=r'E:\videos\GiangHo';print(sum(1 for d in os.listdir(r) if os.path.isdir(os.path.join(r,d)) for f in os.listdir(os.path.join(r,d)) if '_van' in f))"
```

## 7. Những chỗ đã trả giá (đừng lặp lại)

| Sự cố | Nguyên nhân | Đã chặn bằng |
|---|---|---|
| Mốc ván lệch 3-6 phút | nhiều lượt decode ghi chung thư mục frame | thư mục riêng theo tag + PID |
| `Could not open file …jpg` | hai tiến trình cắt cùng lúc xoá frame của nhau | PID trong tên thư mục |
| `CUDA illegal memory access` | job transcribe chiếm hết VRAM | dừng job kia, hoặc `OCR_MODEL_FORMAT=onnx` |
| `Expecting ',' delimiter` | cache bị giết đúng lúc ghi | ghi nguyên tử (.tmp + đổi tên) |
| Mất một ván khi "tối ưu" decode | nới ngưỡng keyframe-only quá tay | ngưỡng `keyframe <= step/8` |
| Nén AV1 chỉ nhỏ đi 2% | encode AV1 sang H.264 là đổi xuôi thành ngược | bỏ qua mọi file bitrate thấp |
