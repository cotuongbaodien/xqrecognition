import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch

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
    board_state = {}
    for box in boxes:
        x, y, w, h = box.xywh[0]
        label = box.cls  # Nhãn quân cờ
        closest = min(grid, key=lambda p: ((p[0]-x)**2 + (p[1]-y)**2)**0.5)
        board_state[label] = board_map[closest]
    return board_state


# Huấn luyện YOLOv8
def train_model():
    model = YOLO("yolov8s.pt")  # Model base
    model.train(
        data="xiangqi_dataset/data.yaml",
        imgsz=416,
        epochs=50,
        batch=8,
        name="xiangqi_model",
        plots=True,
        amp=False
    )
    model.save("xiangqi_yolov8.pt")
    return model

# Phát hiện quân cờ và vị trí


def detect_pieces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh!")
        return

    # Phát hiện lưới bàn cờ
    grid, board_map = detect_board_grid(image)

    # Tải model YOLOv8 đã huấn luyện
    model = YOLO("xiangqi_yolov8.pt")
    results = model(image)

    # Ánh xạ quân cờ
    board_state = map_to_board(results[0].boxes, grid, board_map)

    # In kết quả
    for piece, pos in board_state.items():
        print(f"{piece}: {pos}")

    # Vẽ nhãn lên ảnh
    for box in results[0].boxes:
        x, y, w, h = box.xywh[0]
        label = box.cls
        cv2.putText(
            image,
            f"{label}: {board_map[min(grid, key=lambda p: ((p[0]-x)**2 + (p[1]-y)**2)**0.5)]}",
            (int(x-w/2), int(y-h/2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    cv2.imwrite("output.jpg", image)


# Chạy huấn luyện và phát hiện
if __name__ == "__main__":
    # Bỏ comment để huấn luyện
    train_model()

    # Phát hiện trên ảnh mới
    detect_pieces("input/board.jpg")
