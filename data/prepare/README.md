# Hướng Dẫn Chuẩn Bị Data

## Cấu Trúc Folder

```
data/prepare/
├── images/          ← Đặt ảnh vào đây
│   ├── xxx.png
│   ├── xxx.jpg
│   └── ...
├── labels.csv       ← Edit file này
└── README.md
```

## Bước 1: Thêm Ảnh

Đặt tất cả ảnh vào folder `images/`

**Yêu cầu ảnh:**
- Format: `.png`, `.jpg`, `.jpeg`
- Bàn cờ phải nhìn rõ ràng
- Bàn cờ chiếm > 50% diện tích ảnh
- Không bị crop mất quân

**Nguồn ảnh gợi ý:**
- Screenshots từ app cờ
- Ảnh chụp bàn cờ thật
- Frames từ video YouTube

## Bước 2: Edit labels.csv

Mở file `labels.csv` và thêm dòng cho mỗi ảnh:

```csv
image,fen
ten_anh_1.png,rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
ten_anh_2.jpg,r1bakab1r/9/1cn3nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
```

**Lưu ý:**
- Tên file phải khớp chính xác với ảnh trong `images/`
- FEN phải đúng format (10 hàng, ngăn cách bởi `/`)
- Không có space trong FEN

## FEN Format

```
Row 0 (đen): rnbakabnr    ← Hàng trên cùng
Row 1:       9            ← 9 ô trống
Row 2:       1c5c1        ← 1 trống, pháo, 5 trống, pháo, 1 trống
...
Row 9 (đỏ): RNBAKABNR    ← Hàng dưới cùng
```

**Ký hiệu quân cờ:**

| Quân | Đen (thường) | Đỏ (HOA) |
|------|--------------|----------|
| Xe   | r | R |
| Mã   | n | N |
| Tượng | b | B |
| Sĩ   | a | A |
| Tướng | k | K |
| Pháo | c | C |
| Tốt  | p | P |

## Bước 3: Validate Data

Sau khi chuẩn bị xong, chạy:

```bash
python scripts/auto_learn/validate.py
```

## Ví Dụ

### Thế cờ ban đầu
```
FEN: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
```

### Thế cờ tàn cuộc (chỉ còn 2 tướng)
```
FEN: 4k4/9/9/9/9/9/9/9/9/4K4
```

### Thế cờ giữa ván
```
FEN: r1bakab1r/9/1cn3nc1/p1p1p1p1p/9/4P4/P1P3P1P/1C2C4/9/RNBAKABNR
```

## Checklist Trước Khi Train

- [ ] Có ít nhất 100 ảnh (recommend 500+)
- [ ] Mỗi ảnh có FEN tương ứng trong labels.csv
- [ ] FEN đã kiểm tra đúng
- [ ] Ảnh rõ nét, đủ sáng
- [ ] Mix nhiều nguồn/styles
