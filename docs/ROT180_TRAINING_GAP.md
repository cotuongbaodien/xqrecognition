# Bàn cờ lật ngược (rot180) — lỗ hổng ở TẬP TRAIN, không phải ở code

Status: ~~đã xác định nguyên nhân + tái hiện được, chưa fix~~ (2026-07-20)
→ **2026-08-17: ĐO TRÊN DIỆN RỘNG, giả thuyết bên dưới KHÔNG đứng vững.** Xem
[§Đính chính 2026-08-17](#đính-chính-2026-08-17--đo-trên-4158-ảnh-thật) ở cuối. **Chưa sinh rot180.**

Phân biệt với [`ORIENTATION_FIX.md`](./ORIENTATION_FIX.md): doc kia nói về orientation ở
tầng **hậu xử lý** (canonicalize quad 90°, vote lật dọc trong `detect_board_orientation`).
Doc này nói về tầng **trước đó** — YOLO detector không nhận ra con tướng ngay từ đầu, nên
FEN thiếu hẳn `k`/`K` và pipeline trả `detected=false`. Hậu xử lý có sửa cũng vô ích khi
quân cờ chưa từng được detect.

---

## Triệu chứng thật

Người dùng (code `RDT8-GYA7-N54B-RTFY`) báo 4–5 ảnh liên tiếp không OCR được, ảnh khác
thì bình thường. Log service:

```
13:05:27 detected=False conf=0.907 pieces=14 errors=['missing_black_general']
13:05:34 detected=False conf=0.902 pieces=14 errors=['missing_red_general','missing_black_general']
```

Đáng chú ý: `conf` cao (0.90) và `pieces=14` = **đếm đủ số quân**, nhưng không có con
tướng nào trong FEN. Tức không phải "không thấy quân", mà là **phân loại sai con tướng**.

Chốt chặn nằm ở `boarddetection/server.py:255-268` — FEN thiếu một trong hai tướng thì
ép `detected=False` (bàn không có tướng làm app tính chiếu sai, sự cố 2026-06-12).

## Tái hiện

Gọi thẳng service prod bằng Flow A (`X-OCR-Secret`, không ghi vào ingest queue):

```bash
curl -s -X POST http://127.0.0.1:8001/detect -H "X-OCR-Secret: $S" -F "image=@anh.jpg"
```

| Ảnh | Nguyên bản | Sau biến đổi |
|---|---|---|
| Bàn cờ skin gỗ, viền xanh (đỏ ở TRÊN) | `detected=false`, `missing_black_general` | **rot180 → `detected=true`**, conf 0.887, FEN đủ |
| Screenshot cả màn hình app (1280×591) | `detected=false`, thiếu CẢ HAI tướng | **crop riêng bàn cờ → `detected=true`**, conf 0.88 |

Hai ảnh fail vì hai lý do khác nhau:
- Ảnh bàn cờ: **lật ngược** (帥 đỏ ở trên, 將 đen ở dưới).
- Ảnh screenshot: bàn cờ chỉ chiếm ~48% chiều cao, còn lại là status bar + toolbar +
  panel Engine → không khoanh được vùng bàn.

Lưu ý: screenshot sau khi crop thì đọc đúng ở **cả hai chiều** (ra cùng một FEN). Nghĩa là
pipeline CÓ tự chuẩn hoá hướng, nhưng khả năng đó gãy khi bàn cờ không lấp đầy khung hoặc
tín hiệu màu yếu.

---

## Nguyên nhân gốc

`scripts/train_items.py:63-65`:

```python
flipud=0.0,      # Don't flip vertically (chars would be upside-down)
fliplr=0.5,      # Horizontal flip ok (board is left-right symmetric)
degrees=45.0,    # Rotation up to ±45° (covers most camera tilts)
```

Model **chưa bao giờ** thấy bàn cờ lật ngược: xoay chặn ở ±45°, lật dọc tắt hẳn. Bàn
đỏ-ở-trên nằm hoàn toàn ngoài phân bố train.

### Chỗ suy luận cũ bị hụt

Chú thích "chars would be upside-down" đúng một nửa. Cần tách 3 phép biến đổi:

| Phép | Chữ trông ra sao | Có thật ngoài đời? |
|---|---|---|
| `flipud` đơn lẻ | soi gương dọc | ❌ không bao giờ |
| `fliplr` đơn lẻ | soi gương ngang | ❌ không bao giờ |
| **rot180** = flipud + fliplr | ngược đầu, KHÔNG soi gương | ✅ ngồi phía đối diện bàn |

Tắt `flipud` riêng lẻ là hợp lý. Nhưng rot180 cần **cả hai** bật cùng lúc, nên tắt một cái
đã chặn luôn trường hợp thật duy nhất.

Hệ quả kèm theo đáng nghi: `fliplr=0.5` đang bật đơn lẻ → 50% ảnh train có chữ **soi gương
ngang**, thứ không tồn tại ngoài đời, mà lại đúng là tín hiệu phân biệt 帥 với 將.

---

## rot180 an toàn về nhãn — đã kiểm chứng

Sinh dữ liệu rot180 từ tập đã gán nhãn là **biến đổi toạ độ thuần tuý**:

```
x → 1 - x      y → 1 - y      class giữ nguyên      w, h giữ nguyên
```

Điểm dễ vỡ nhất là class `palace-bottom` (id 8) — nghe như phụ thuộc hướng. Đã kiểm tra
`boarddetection/item_detector.py:533`:

```python
"palace-bottom": [(3, 0), (5, 0), (3, 9), (5, 9)],
```

Nó được định nghĩa ở **cả hai đầu bàn**, nên qua rot180 tập class ánh xạ về chính nó.
**Không cần remap class nào.** 17 class còn lại bất biến (帥 xoay ngược vẫn là 帥).

→ 6.107 ảnh train + 1.229 val trong `data/items_v20` nhân đôi được ngay, không phải chụp
lại hay gán nhãn lại.

---

## Kế hoạch sửa (chưa làm)

**1. Sinh offline, KHÔNG dùng `degrees=180`.**
Chỉnh `degrees` lên 180 chỉ tốn một dòng nhưng rải đều mọi góc — kể cả 60–120° không xảy ra
ngoài đời — làm loãng dữ liệu. Sinh cứng bản rot180 cho từng ảnh đã gán nhãn thì đúng phân
bố thật và kiểm soát được.

**2. Đo baseline TRƯỚC khi train.**
rot180 tập đã gán nhãn rồi chấm bằng model hiện tại → ra ngay con số hổng. Không có baseline
thì train xong không biết có đỡ thật không.

**3. Đánh giá bằng bộ real-world, ĐỪNG tin val mAP.**
Bài học đã ghi sẵn trong `train_items.py:57-61`: v10 augment nhẹ hơn cho val mAP tốt hơn
(0.80 vs 0.78) nhưng thực tế **tệ hơn hẳn** (EXACT 7/28 so với 11/28 của v9). Nhớ thêm ảnh
lật ngược + skin gỗ vào bộ 28 ảnh đó và vào `test/`.

**4. Thí nghiệm riêng cho `fliplr`.**
Một nhánh `fliplr=0.0` + rot180 offline, so với nhánh giữ nguyên. Nghi ngờ soi-gương-ngang
làm hại việc phân biệt tướng — nhưng đó là giả thuyết, phải đo.

**5. Regression guard:** rerun eval 86 ảnh (`detect.py` + `eval_fen.py`), không được tụt
dưới mức tốt nhất hiện tại (v15 58 / v14 61).

---

## Lỗ hổng thứ hai — skin gỗ, quân đỏ không có mực đỏ

Ảnh bàn cờ fail ở trên dùng skin gỗ: quân đỏ và quân đen **gần như cùng màu**, chỉ khác mặt
chữ. Mất tín hiệu màu — mà theo `ORIENTATION_FIX.md` §Problem 2, màu chính là tín hiệu
robust nhất để quyết định palace nào là đỏ.

Xoay dữ liệu **không** cứu được ca này; cần đa dạng skin. Đây mới là chỗ generator dữ liệu
trả công xứng đáng (đã có tiền lệ `data/board_seg_v6_synth500`): render bàn tổng hợp theo
tổ hợp **skin × hướng × góc nhìn**.

Thứ tự đề xuất: script rot180 + đo baseline trước → có số rồi mới quyết mức đầu tư cho
generator skin.

---

## Việc cần làm riêng, ngoài phạm vi train

Ảnh screenshot fail vì bàn cờ không lấp đầy khung. Fix code-only, không cần train lại:
thêm bước **tự dò và crop khung bàn cờ** trước khi infer, và/hoặc thử cả 2 chiều rồi lấy
kết quả có đủ 2 tướng. Chặn được cả hai ca trong doc này mà không đụng tới model.

---

## Đính chính 2026-08-17 — đo trên 4158 ảnh thật

Doc trên suy nguyên nhân **từ config aug** (`flipud=0.0` → "model chưa bao giờ thấy bàn lật
ngược"), chứ không đo phân bố dữ liệu. Đo rồi thì suy luận đó sai ở tiền đề.

**1. Bàn lật ngược KHÔNG hiếm trong train.** Đo hướng bằng centroid y của quân đỏ so với
quân đen trên chính file nhãn:

| Tập | tỉ lệ bàn lật (đen ở dưới) |
|---|---|
| ảnh khách gửi kỳ 08 (3635 đo được) | 44,5% |
| ảnh khách gửi kỳ 07 (2320) | 44,4% |
| **`items_v20/train` — ảnh thật (2331)** | **37,5%** |
| `items_v20/train` — ảnh synth (3087) | 34,6% |

`flipud` không liên quan: bản thân ảnh khách gửi đã có sẵn ~44% bàn lật và chúng đã nằm
trong train. Aug chỉ cần thiết khi phân bố thật thiếu — ở đây không thiếu.

**2. Prod không fail nhiều hơn ở bàn lật.** Đối chiếu 3808 ảnh kỳ 08 với cột `status` thật
trong `ocr_logs` (kết quả prod trả về lúc khách gửi):

| hướng bàn | n | prod trả `undetected` |
|---|---|---|
| đỏ dưới (chuẩn) | 2019 | 3,96% |
| đen dưới (lật) | 1616 | 5,01% |
| model chỉ thấy lác đác quân (<4 quân/bên) | 173 | **23,12%** |

Chênh 1,05 điểm % giữa hai hướng → **z≈1,5, p≈0,13: nằm trong nhiễu.** Chỗ mất ảnh thật sự
là nhóm thứ ba, fail gấp ~5 lần — đúng hai ca doc mô tả (screenshot bàn không lấp đầy khung,
skin gỗ mất tín hiệu màu). Tức **lỗ hổng thứ hai của doc mới là cái đáng đầu tư**, không
phải rot180.

**3. Giới hạn của phép đo này — và việc phải làm tiếp.** `ocr_logs` chỉ ghi
`detected`/`undetected`, **không biết FEN đúng hay sai**. Cảm nhận "bàn lật đọc dở" của
người dùng nhiều khả năng là **FEN sai quân** (prod vẫn trả `detected=true`), loại lỗi mà
số liệu trên hoàn toàn không chạm tới.

→ Bench holdout 700 (kỳ 2026-08-17) đã tách **cân bằng hướng: 343 đỏ-dưới / 307 đen-dưới /
50 khó**, đủ để chấm tách hai nhóm. **Nhập FEN GT xong, chấm tách hướng — có số rồi mới
quyết có sinh rot180 hay không.** Sinh rot180 bây giờ là nhân đôi 6107 ảnh train dựa trên
một giả thuyết đã bị số liệu phản bác.

> Ghi chú thứ tự: nếu có sinh rot180 thì phải sinh **SAU** khi review/sửa nhãn xong, nếu
> không mỗi ô sai xuất hiện 2 lần trong gallery (sửa gấp đôi) và nhãn rác cũng bị nhân đôi.

---

## Đo tiếp 2026-08-17 — tướng biến mất vì ĐÂU? (và một fix code đã làm)

Nghi vấn của người dùng: *"có khi nhận tướng nhầm khung nên bỏ"*. Cơ chế đó **có thật**
trong code — `rules_validator.py::_fix_invalid_positions` xoá quân nằm ngoài
`VALID_POSITIONS` khi conf < 0.7, và tướng không hề được miễn trừ. Nhưng đo rồi thì nó
**không phải thủ phạm chính**.

Chạy pipeline thật + instrument (đếm quân bị validator xoá, ghi lại phiếu bầu hướng) trên
201 ảnh prod ĐÃ trả `undetected` + 250 ảnh prod trả `detected`:

| | 201 ảnh prod fail | 250 ảnh prod OK |
|---|---|---|
| kết quả thiếu tướng | 135 | 8 |
| → **do validator xoá tướng** | **13 (10%)** | 1 |
| → do detector không thấy tướng | **122 (90%)** | 7 |

Và **không phải lỗi hướng**: 13/14 ca bỏ phiếu ra `standard`; conf của các tướng bị xoá
0.26–0.67 (model vốn đã không chắc). Phép thử bất biến rot180 (chạy pipeline 2 chiều,
so FEN) trên 279 ảnh: **0 ca** nào có `FEN(xoay) == rot180(FEN(gốc))` → tầng chuẩn hoá
hướng ở `pipeline.py:223-229` **luôn hoạt động**. Khác biệt giữa 2 chiều (29,6% ở nhóm
detected) là do **detector đọc ra bộ quân khác**, không phải bàn bị để lật.

> rot180 cứu được 45/201 ảnh fail — nhưng **42/45 là do detector đọc được ở chiều kia**,
> chỉ 3 do validator. Nút thắt nằm ở **detector không nhận ra con tướng**, tức 2 lớp
> `soaiden`/`soaido` trong vòng review nhãn — không phải ở hậu xử lý.

### Fix đã làm: tướng được miễn xoá, snap về ô khung gần nhất

Tướng là quân **duy nhất** mỗi bên và `server.py:255-268` ép `detected=False` khi thiếu →
xoá tướng biến "bàn sai một phần" thành "hỏng chắc chắn". Tướng nằm ô bất hợp lệ là lỗi
map lưới, không phải quân ma. `rules_validator.py` nay:
- `K`/`k` **không bị xoá**; gọi `_snap_general()` → dời sang ô khung TRỐNG gần nhất
  (khoảng cách lưới bình phương, tie-break `(row, col)` cho tất định).
- Khung kín hết chỗ → giữ nguyên tại chỗ (ô bất hợp lệ vẫn hơn mất tướng).
- Các quân khác giữ nguyên hành vi cũ.

Đo lại trên đúng bộ ảnh trên:

| | trước | sau |
|---|---|---|
| 201 ảnh prod fail | 135 thiếu tướng | **cứu 11, hỏng 0** |
| 250 ảnh prod OK | — | **hỏng 0** (1 ảnh đổi FEN: tướng snap về ô hợp lệ) |
| **bench 243 (FEN gate)** | **229/243** | **229/243 — không đổi** |

Chỉ sửa hậu xử lý Python, **không đụng weights → không cần export lại ONNX**; deploy chỉ
cần ship `boarddetection/rules_validator.py` + restart container.
