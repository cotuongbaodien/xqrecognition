#!/usr/bin/env python3
import os
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import shutil
import yaml
import numpy as np

base_dir = Path(__file__).parent.parent
BOARD_IMG = f"{base_dir}/scripts/pieces/board.jpg"
PIECES_DIR = f"{base_dir}/scripts/pieces"
OUTPUT_DIR = f"{base_dir}/scripts/enhanced_dataset"
NUM_TRAIN = 200  # Tăng số lượng
NUM_VAL = 50
NUM_TEST = 20

# Load class names
CLASS_NAMES = {}
with open(f"{base_dir}/dataset/data.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    CLASS_NAMES.update({i: name for i, name in enumerate(data["names"]) if name != "river"})

ROWS = 10
COLS = 9

def make_dirs():
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

def get_board_positions(board_w, board_h):
    """Tạo grid positions cho bàn cờ"""
    cell_w = board_w / (COLS - 1)
    cell_h = board_h / (ROWS - 1)
    positions = []
    for r in range(ROWS):
        for c in range(COLS):
            x = c * cell_w
            y = r * cell_h
            positions.append((x, y))
    return positions

def create_game_scenarios():
    """Tạo các tình huống game thực tế"""
    scenarios = [
        # Khai cuộc - nhiều quân
        {
            "name": "opening",
            "pieces": [
                "Red General", "Red Advisor", "Red Elephant", "Red Horse", "Red Chariot", "Red Cannon", "Red Soldier",
                "Black General", "Black Advisor", "Black Elephant", "Black Horse", "Black Chariot", "Black Cannon", "Black Soldier"
            ],
            "min_pieces": 12,
            "max_pieces": 16
        },
        # Trung cuộc - ít quân hơn
        {
            "name": "middlegame", 
            "pieces": [
                "Red General", "Red Advisor", "Red Horse", "Red Chariot", "Red Cannon", "Red Soldier",
                "Black General", "Black Advisor", "Black Horse", "Black Chariot", "Black Cannon", "Black Soldier"
            ],
            "min_pieces": 8,
            "max_pieces": 12
        },
        # Tàn cuộc - rất ít quân
        {
            "name": "endgame",
            "pieces": [
                "Red General", "Red Advisor", "Red Soldier",
                "Black General", "Black Advisor", "Black Soldier"
            ],
            "min_pieces": 4,
            "max_pieces": 8
        }
    ]
    return scenarios

def enhanced_augmentation(piece_img):
    """Cải thiện data augmentation"""
    # Xoay
    angle = random.uniform(-20, 20)
    piece_img = piece_img.rotate(angle, expand=True)
    
    # Scale
    scale = random.uniform(0.7, 1.3)
    new_size = (int(piece_img.width * scale), int(piece_img.height * scale))
    piece_img = piece_img.resize(new_size, Image.LANCZOS)
    
    # Color adjustments
    enhancer = ImageEnhance.Brightness(piece_img)
    piece_img = enhancer.enhance(random.uniform(0.6, 1.4))
    
    enhancer = ImageEnhance.Contrast(piece_img)
    piece_img = enhancer.enhance(random.uniform(0.6, 1.4))
    
    enhancer = ImageEnhance.Color(piece_img)
    piece_img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Thêm noise nhẹ
    if random.random() < 0.3:
        piece_img = add_noise(piece_img)
    
    return piece_img

def add_noise(img, intensity=0.02):
    """Thêm noise vào ảnh"""
    img_array = np.array(img)
    noise = np.random.normal(0, intensity * 255, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def add_environmental_effects(board):
    """Thêm hiệu ứng môi trường"""
    # Blur nhẹ
    if random.random() < 0.2:
        board = board.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    # Thay đổi màu sắc
    if random.random() < 0.3:
        enhancer = ImageEnhance.Color(board)
        board = enhancer.enhance(random.uniform(0.8, 1.2))
    
    return board

def generate_realistic_image(index, split, positions, scenarios):
    """Tạo ảnh với tình huống game thực tế"""
    board = Image.open(BOARD_IMG).convert("RGBA")
    board_w, board_h = board.size
    
    # Chọn scenario ngẫu nhiên
    scenario = random.choice(scenarios)
    num_pieces = random.randint(scenario["min_pieces"], scenario["max_pieces"])
    
    # Chọn quân cờ từ scenario
    available_pieces = scenario["pieces"].copy()
    chosen_pieces = random.sample(available_pieces, min(num_pieces, len(available_pieces)))
    
    # Chọn vị trí
    chosen_positions = random.sample(positions, len(chosen_pieces))
    
    label_lines = []
    for i, (piece_name, pos) in enumerate(zip(chosen_pieces, chosen_positions)):
        # Tìm class_id
        class_id = None
        for cid, name in CLASS_NAMES.items():
            if name == piece_name:
                class_id = cid
                break
        
        if class_id is not None:
            piece_file = f"{PIECES_DIR}/{piece_name}.png"
            if os.path.exists(piece_file):
                piece_img = Image.open(piece_file).convert("RGBA")
                piece_img = enhanced_augmentation(piece_img)
                
                # Tính toán YOLO format
                x_center = pos[0] / board_w
                y_center = pos[1] / board_h
                w = piece_img.width / board_w
                h = piece_img.height / board_h
                
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    # Thêm hiệu ứng môi trường
    board = add_environmental_effects(board)
    
    # Lưu ảnh và label
    img_path = f"{OUTPUT_DIR}/{split}/images/{index}.jpg"
    board.convert("RGB").save(img_path, quality=95)
    
    label_path = f"{OUTPUT_DIR}/{split}/labels/{index}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    make_dirs()
    
    temp_board = Image.open(BOARD_IMG)
    positions = get_board_positions(temp_board.width, temp_board.height)
    scenarios = create_game_scenarios()
    
    print("Generating training images...")
    for i in range(NUM_TRAIN):
        generate_realistic_image(i, "train", positions, scenarios)
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{NUM_TRAIN} training images")
    
    print("Generating validation images...")
    for i in range(NUM_VAL):
        generate_realistic_image(i, "valid", positions, scenarios)
    
    print("Generating test images...")
    for i in range(NUM_TEST):
        generate_realistic_image(i, "test", positions, scenarios)
    
    # Tạo classes.txt
    with open(f"{OUTPUT_DIR}/classes.txt", "w") as f:
        f.write("\n".join(CLASS_NAMES.values()))
    
    # Tạo data.yaml mới
    data_yaml = {
        'train': './train/images',
        'val': './valid/images', 
        'test': './test/images',
        'nc': len(CLASS_NAMES),
        'names': list(CLASS_NAMES.values())
    }
    
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Enhanced dataset generated!")
    print(f"Training images: {NUM_TRAIN}")
    print(f"Validation images: {NUM_VAL}")
    print(f"Test images: {NUM_TEST}")
    print(f"Total images: {NUM_TRAIN + NUM_VAL + NUM_TEST}")

if __name__ == "__main__":
    main()
