# Phân Tích Domain Gap - Model Piece Detection

## Vấn Đề

Model `pieces_det.pt` (YOLOv8n) được train trên dataset `pieces/` chỉ có ảnh screenshot app cờ tướng online. Khi dùng trên ảnh chụp bàn cờ thật, model hoạt động rất kém.

## Thực Nghiệm

### Test 1: Ảnh app (test/1.png)
- **Kết quả:** 32/32 quân detected, confidence 0.95-0.99
- **FEN:** Chính xác
- **Nhận xét:** Model hoạt động tốt trên domain đã train

### Test 2: Ảnh thật (fendata/002.png)
- **Kết quả:** 28 quân detected (conf>=0.3), nhiều sai class
- **Lỗi cụ thể:**
  - 8x Elephant_red (B) - max hợp lệ là 2
  - Cannon (c/C) bị nhầm thành Elephant (b/B)
  - Pawn (p/P) bị nhầm thành Elephant (b/B)
  - Rook (r/R) bị miss hoặc sai class
- **FEN accuracy:** 0%

### Test 3: 20 ảnh fendata
- **FEN exact match:** 0/20 = 0%
- **Mẫu sai chung:** Predictions chỉ có 8-12 quân đúng, phần còn lại sai class hoặc miss

## Nguyên Nhân

### 1. Visual Domain Khác Biệt

| Đặc điểm | App (train) | Thật (test) |
|----------|-------------|-------------|
| Màu quân | Vàng/nâu, có hiệu ứng 3D | Trắng/kem, phẳng |
| Chữ Hán | Lớn, rõ nét, font đều | Nhỏ hơn, viết tay, khác font |
| Màu phân biệt đỏ/đen | Màu sắc rõ ràng | Chỉ khác màu chữ (đỏ/đen) |
| Background | Nền cố định, sạch | Bàn cờ thật, có vết bẩn, bóng |
| Góc chụp | Luôn thẳng đứng | Có góc nghiêng, perspective |
| Ánh sáng | Đồng nhất | Không đều, có bóng đổ |

### 2. Feature Distribution Shift

Model đã học các feature đặc trưng của app:
- Gradient màu vàng/nâu của quân cờ app
- Viền tròn có hiệu ứng bóng
- Font chữ cố định

Các feature này KHÔNG tồn tại trong ảnh bàn cờ thật -> model không nhận diện được.

### 3. Class Confusion Pattern

```
Ảnh thật -> Model predict:
  Cannon (C/c) -> Elephant (B/b)   [RẤT THƯỜNG]
  Pawn (P/p)   -> Elephant (B/b)   [THƯỜNG]
  Rook (R/r)   -> Knight (N/n)     [THẤP]
  Advisor (A/a) -> General (K/k)   [TRUNG BÌNH]
```

Lý do: Trong ảnh thật, các quân cờ tròn trắng giống nhau về hình dạng, chỉ khác chữ Hán. Model nano không đủ capacity để học phân biệt chữ Hán.

## Giải Pháp

### Ngắn hạn: Retrain với pieces_merged/
- Dataset đã có 1431 ảnh train bao gồm cả ảnh thật
- Dùng model lớn hơn (YOLOv8s)
- Thêm augmentation: perspective, rotation, lighting

### Dài hạn: Auto-learning pipeline
- Dùng prepare/ (786 ảnh có FEN label)
- Tự động tạo YOLO labels từ FEN + board detection
- Liên tục bổ sung ảnh mới từ nhiều nguồn

## Kết Luận

Đây là vấn đề **data distribution**, không phải vấn đề **code/pipeline**. Các fix pipeline (NMS, collision, rules) vẫn cần thiết nhưng chỉ hiệu quả khi model detect đúng class. Ưu tiên #1 là retrain model với dataset đa dạng hơn.
