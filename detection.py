import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import shutil

# Kiểm tra GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, using CPU")

# Tạo cấu trúc dataset
dataset_dir = "xiangqi_dataset"
os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)


# Hàm phát hiện lưới bàn cờ 9x10
def detect_board_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Tăng độ tương phản
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Phát hiện đường thẳng bằng Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=100, minLineLength=100, maxLineGap=10)

    # Phân loại đường ngang và dọc
    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) > abs(y1 - y2):  # Đường ngang
            h_lines.append((y1 + y2) / 2)
        else:  # Đường dọc
            v_lines.append((x1 + x2) / 2)

    # Lấy 10 đường ngang và 9 đường dọc
    h_lines = sorted(h_lines)[:10]
    v_lines = sorted(v_lines)[:9]

    # Tạo lưới giao điểm 9x10
    grid = []
    board_map = {}
    for i, y in enumerate(h_lines):
        for j, x in enumerate(v_lines):
            grid.append((x, y))
            board_map[(x, y)] = f"{chr(97+j)}{9-i}"  # a0-i9
    return grid, board_map


# Hàm ánh xạ quân cờ vào lưới
def map_to_board(boxes, grid, board_map):
    # Xử lý trùng lặp vị trí - chỉ giữ lại quân có độ tin cậy cao nhất ở mỗi vị trí
    positions_seen = {}  # Lưu vị trí đã gặp và confidence cao nhất
    
    for box in boxes:
        x, y, w, h = box.xywh[0]
        label_idx = int(box.cls)  # Index của quân cờ
        confidence = float(box.conf)  # Độ tin cậy của dự đoán
        closest = min(grid, key=lambda p: ((p[0]-x)**2 + (p[1]-y)**2)**0.5)
        position = board_map[closest]
        
        # Nếu vị trí đã có quân khác và quân hiện tại có độ tin cậy thấp hơn, bỏ qua
        if position in positions_seen and positions_seen[position]['confidence'] > confidence:
            continue
            
        # Lưu hoặc cập nhật quân cờ ở vị trí này
        positions_seen[position] = {
            'piece': label_idx,
            'position': position,
            'confidence': confidence
        }
    
    # Chuyển dict thành list và sắp xếp theo độ tin cậy giảm dần
    board_state = list(positions_seen.values())
    return sorted(board_state, key=lambda x: (-x['confidence'], x['position']))


# Huấn luyện YOLOv8
def train_model():
    model = YOLO("yolov8m.pt")  # Sử dụng model medium để có độ chính xác cao hơn
    model.train(
        data="chinese-chess-detect-for-yolo-8/data.yaml",  # Dataset
        imgsz=640,               # Kích thước phù hợp với quân cờ
        epochs=150,              # Tăng số epoch để học tốt hơn
        batch=8,                 # Giảm batch size cho RTX 3060
        name="xiangqi_model",    # Tên thư mục output
        workers=2,               # Giảm số luồng để tiết kiệm RAM
        patience=30,             # Tăng patience để tránh dừng sớm
        optimizer='AdamW',       # Dùng AdamW cho hiệu quả tốt hơn
        lr0=0.001,              # Learning rate khởi đầu ổn định
        lrf=0.0001,             # Learning rate cuối thấp hơn
        weight_decay=0.0005,     # Giữ regularization
        warmup_epochs=5,         # Tăng warmup để ổn định
        box=7.0,                 # Tăng box loss vì vị trí quan trọng
        cls=1.0,                 # Tăng class loss
        hsv_h=0.2,              # Giảm hue vì màu sắc quan trọng
        hsv_s=0.5,              # Giảm saturation
        hsv_v=0.3,              # Giảm value
        degrees=15.0,           # Tăng góc xoay
        translate=0.3,          # Tăng mức dịch chuyển
        scale=0.4,              # Tăng scale
        shear=0.2,              # Tăng shear
        perspective=0.0,        # Không dùng perspective
        flipud=0.0,             # Không lật dọc
        fliplr=0.3,             # Lật ngang vừa phải
        mosaic=0.7,             # Giảm mosaic
        mixup=0.1,              # Thêm mixup nhẹ
        amp=False,              # Tắt mixed precision do lỗi CUDA
        plots=True              # Lưu biểu đồ loss
    )
    
    # Check if the weights file exists before copying
    weights_path = "runs/detect/xiangqi_model/weights/best.pt"
    if os.path.exists(weights_path):
        print(f"Copying best weights from {weights_path} to xiangqi_yolov8.pt")
        shutil.copy(weights_path, "xiangqi_yolov8.pt")
    else:
        print(f"Warning: Weights file not found at {weights_path}")
        print("Training may not have completed successfully")
    
    return model

# Phát hiện quân cờ và vị trí


def detect_pieces(image_path):
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh!")
        return
        
    # In thông tin về ảnh
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Pixel value range: [{np.min(image)}, {np.max(image)}]")
    
    # Kiểm tra và chuẩn hóa ảnh
    if len(image.shape) != 3:
        print("Warning: Image is not in RGB format")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] != 3:
        print("Warning: Unexpected number of channels")
    
    # Phát hiện lưới bàn cờ
    grid, board_map = detect_board_grid(image)

    # Tải model YOLOv8 đã huấn luyện
    print("Loading model...")
    model_path = "xiangqi_yolov8.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    try:
        # Load model với half precision để giảm VRAM và tăng tốc độ
        model = YOLO(model_path)
        model.to('cuda')  # Đảm bảo model ở trên GPU
        if torch.cuda.is_available():
            model.model.half()  # Chuyển sang FP16 để tăng tốc độ
        print("Model loaded successfully")
        # In thông tin về model
        print(f"Model info: {model.info()}")
        print(f"Model names (classes): {model.names}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
        
    try:
        # Tiền xử lý ảnh để tăng độ tương phản và độ rõ nét
        all_detections = []
        
        # Các kích thước ảnh khác nhau để thử
        image_sizes = [640, 800, 1024]
        
        # Các phiên bản xử lý ảnh khác nhau
        processed_images = []
        
        # 1. Ảnh gốc
        processed_images.append(image.copy())
        
        # 2. Tăng độ tương phản mạnh hơn
        contrast_enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=10)
        processed_images.append(contrast_enhanced)
        
        # 3. Cân bằng màu với CLAHE mạnh hơn
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        color_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 4. Tăng độ sắc nét
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        processed_images.append(sharpened)
        
        # 5. Kết hợp CLAHE với sharpening
        sharp_enhanced = cv2.filter2D(color_enhanced, -1, kernel)
        processed_images.append(color_enhanced)
        
        # Thử với mỗi phiên bản ảnh và mỗi kích thước
        for img in processed_images:
            for size in image_sizes:
                # Resize giữ tỷ lệ
                height, width = img.shape[:2]
                scale = size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_resized = cv2.resize(img, (new_width, new_height))
                
                # Pad ảnh
                pad_h = size - new_height
                pad_w = size - new_width
                image_padded = cv2.copyMakeBorder(image_resized, 0, pad_h, 0, pad_w,
                                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
                
                # Inference với các tham số
                with torch.amp.autocast('cuda'):
                    results = model(image_padded,
                                  conf=0.15,    # Giảm ngưỡng confidence để bắt được nhiều quân hơn
                                  iou=0.45,     # Tăng IoU để lọc trùng lặp tốt hơn
                                  max_det=32,   # Số quân cờ tối đa
                                  agnostic_nms=False,  # Tắt agnostic_nms để phân biệt các class
                                  half=True)    # Sử dụng FP16
                    
                # Thêm detection với điều kiện phù hợp
                for box in results[0].boxes:
                    conf = float(box.conf)
                    if conf >= 0.15:  # Chỉ lấy các detection có độ tin cậy đủ cao
                        # Kiểm tra xem detection này có trùng với detection nào trước đó không
                        x, y, w, h = box.xywh[0]
                        is_duplicate = False
                        for existing_box in all_detections:
                            ex_x, ex_y, ex_w, ex_h = existing_box.xywh[0]
                            # Tính khoảng cách giữa tâm của 2 box
                            distance = ((x - ex_x) ** 2 + (y - ex_y) ** 2) ** 0.5
                            if distance < min(w, h) * 0.5:  # Nếu quá gần nhau
                                if conf <= float(existing_box.conf):  # Và confidence thấp hơn
                                    is_duplicate = True
                                    break
                        
                        if not is_duplicate:
                            all_detections.append(box)
                
        # Gộp và lọc kết quả
        results = [results[0]]  # Giữ lại kết quả cuối để dùng cho phần sau
        results[0].boxes = all_detections
        
        # In tóm tắt kết quả
        n_detections = len(all_detections)
        print(f"\nTìm thấy {n_detections} quân cờ")
        
        if len(results[0].boxes) == 0:
            # Thử detect với ảnh gốc
            print("Trying detection with original image...")
            results = model(image, conf=0.05)
            print(f"Raw detection results (original size): {results[0]}")
            
        if len(results[0].boxes) == 0:
            print("No objects detected. Try adjusting confidence threshold.")
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        return
    # Dictionary chuyển đổi tên quân cờ
    piece_names = {
        'b_shi': 'Sĩ Đen',
        'b_pao': 'Pháo Đen',
        'b_ju': 'Xe Đen',
        'b_xiang': 'Tượng Đen',
        'b_jiang': 'Tướng Đen',
        'b_ma': 'Mã Đen',
        'b_zu': 'Tốt Đen',
        'r_shi': 'Sĩ Đỏ',
        'r_pao': 'Pháo Đỏ',
        'r_ju': 'Xe Đỏ',
        'r_xiang': 'Tượng Đỏ',
        'r_jiang': 'Tướng Đỏ',
        'r_ma': 'Mã Đỏ',
        'r_bing': 'Binh Đỏ'
    }

    print(f"\nDetection results: {len(results[0].boxes)} objects found")
    print("-" * 50)

    # Ánh xạ quân cờ
    board_state = map_to_board(results[0].boxes, grid, board_map)
    
    # In kết quả phân tích và tổng hợp theo màu
    red_pieces = 0
    black_pieces = 0
    for piece in board_state:
        piece_code = model.names[piece['piece']]
        piece_name = piece_names[piece_code]
        confidence = piece['confidence'] * 100
        position = piece['position']
        print(f"{piece_name:<12} tại {position:<4} (độ tin cậy: {confidence:>5.1f}%)")
        
        # Đếm số quân theo màu
        if piece_code.startswith('r_'):
            red_pieces += 1
        else:
            black_pieces += 1
            
    print(f"\nTổng số quân cờ: {len(board_state)}")
    print(f"- Quân đỏ: {red_pieces}")
    print(f"- Quân đen: {black_pieces}")

    # Vẽ nhãn lên ảnh
    for box in results[0].boxes:
        x, y, w, h = box.xywh[0]
        piece_code = model.names[int(box.cls)]  # Lấy mã quân cờ
        piece_type = piece_names[piece_code]    # Chuyển đổi sang tên tiếng Việt
        position = board_map[min(grid, key=lambda p: ((p[0]-x)**2 + (p[1]-y)**2)**0.5)]
        
        # Chọn màu dựa vào loại quân (đỏ hoặc đen)
        color = (0, 0, 255) if piece_type.startswith('r_') else (0, 255, 0)
        
        # Vẽ bounding box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ nhãn với font lớn hơn và có nền
        label = f"{piece_type}: {position}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Tính kích thước text để vẽ nền
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        
        # Vẽ text
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),  # Màu trắng cho text
            thickness
        )
    cv2.imwrite("output.jpg", image)


# Chạy huấn luyện và phát hiện
if __name__ == "__main__":
    # Bỏ comment để huấn luyện
    # train_model()

    # Phát hiện trên ảnh mới
    detect_pieces("testdata/test10.png")
    
