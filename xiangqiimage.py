import cv2
import numpy as np
from ultralytics import YOLO

# Danh sách class theo đúng thứ tự trong dataset
class_names = [
    'black-general', 'black-chariot', 'black-horse', 'black-cannon',
    'black-elephant', 'black-advisor', 'black-soldier',
    'red-general', 'red-chariot', 'red-horse', 'red-cannon',
    'red-elephant', 'red-advisor', 'red-soldier'
]

# Mapping class → ký hiệu FEN
fen_map = {
    'black-general': 'k', 'black-chariot': 'r', 'black-horse': 'n',
    'black-cannon': 'c', 'black-elephant': 'b', 'black-advisor': 'a', 'black-soldier': 'p',
    'red-general': 'K', 'red-chariot': 'R', 'red-horse': 'N',
    'red-cannon': 'C', 'red-elephant': 'B', 'red-advisor': 'A', 'red-soldier': 'P'
}

def get_board_grid(img_shape):
    """Chia bàn cờ thành lưới 9x10."""
    h, w = img_shape[:2]
    cell_w = w / 9
    cell_h = h / 10
    grid = []
    for row in range(10):
        for col in range(9):
            x_center = col * cell_w + cell_w / 2
            y_center = row * cell_h + cell_h / 2
            grid.append((col, row, x_center, y_center))
    return grid

def find_closest_cell(x, y, grid):
    """Tìm ô gần nhất để gán quân."""
    min_dist = float('inf')
    best_cell = (0, 0)
    for col, row, cx, cy in grid:
        dist = (x - cx)**2 + (y - cy)**2
        if dist < min_dist:
            min_dist = dist
            best_cell = (col, row)
    return best_cell

def convert_to_fen(board):
    """Chuyển ma trận 2D thành FEN."""
    fen = ''
    for row in board:
        empty = 0
        for cell in row:
            if cell == '':
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += cell
        if empty > 0:
            fen += str(empty)
        fen += '/'
    return fen.rstrip('/')

def detect_and_convert(image_path, model_path='best.pt'):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    results = model(image)[0]

    grid = get_board_grid(image.shape)
    board = [['' for _ in range(9)] for _ in range(10)]

    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        label = class_names[int(cls)]
        piece = fen_map[label]

        col, row = find_closest_cell(x, y, grid)
        board[row][col] = piece

    fen = convert_to_fen(board)
    return fen
if __name__ == "__main__":
    # Bỏ comment để huấn luyện
    # train_model()

    # Phát hiện trên ảnh mới
    fen = detect_and_convert("testdata/test10.png")
    print("FEN:", fen)
