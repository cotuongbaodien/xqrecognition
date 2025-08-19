#!/usr/bin/env python3
import os
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import shutil
import yaml
import numpy as np
import cv2

root_dir = Path(__file__).parent.parent
base_dir = root_dir / "scripts" / "simulate"
BOARD_IMG = base_dir / "pieces" / "board.jpg"
PIECES_DIR = base_dir / "pieces"
OUTPUT_DIR = base_dir / "output"
NUM_TRAIN = 100
NUM_VAL = 10
NUM_TEST = 10

CLASS_NAMES = {}
with open(root_dir / "dataset" / "data.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    CLASS_NAMES.update({i: name for i, name in enumerate(data["names"])})

# board matrix
ROWS = 10
COLS = 9


def make_dirs():
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)


def detect_board_boundaries(board_img):
    board_cv = cv2.cvtColor(np.array(board_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(board_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(th, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found, using the entire image")
        return 0, 0, board_img.width, board_img.height
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    board_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.1 * board_img.width * board_img.height:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            board_contour = approx
            break
    if board_contour is None:
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"No 4 corners found, using bounding rect: {x},{y},{w},{h}")
        return x, y, w, h
    x, y, w, h = cv2.boundingRect(board_contour)
    print(f"Detected board at: {x},{y},{w},{h}")
    return x, y, w, h


def get_board_positions(board_img):
    board_x, board_y, board_w, board_h = detect_board_boundaries(board_img)
    print(f"Board boundaries: x={board_x}, y={board_y}, w={board_w}, h={board_h}")
    cell_w = board_w / COLS
    cell_h = board_h / ROWS
    print(f"Cell size: {cell_w:.1f} x {cell_h:.1f}")
    positions = [(board_x + c * cell_w + cell_w / 2, board_y + r * cell_h + cell_h / 2) for r in range(ROWS) for c in range(COLS)]
    return positions


def create_game_scenarios():
    scenarios = [
        {
            "name": "opening",
            "pieces": [
                "Red General", "Red Advisor", "Red Elephant", "Red Horse", "Red Chariot", "Red Cannon", "Red Soldier",
                "Black General", "Black Advisor", "Black Elephant", "Black Horse", "Black Chariot", "Black Cannon", "Black Soldier"
            ],
            "min_pieces": 12,
            "max_pieces": 16
        },
        {
            "name": "middlegame",
            "pieces": [
                "Red General", "Red Advisor", "Red Horse", "Red Chariot", "Red Cannon", "Red Soldier",
                "Black General", "Black Advisor", "Black Horse", "Black Chariot", "Black Cannon", "Black Soldier"
            ],
            "min_pieces": 8,
            "max_pieces": 12
        },
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
    angle = random.uniform(-20, 20)
    piece_img = piece_img.rotate(angle, expand=True)
    scale = random.uniform(0.7, 1.3)
    new_size = (int(piece_img.width * scale), int(piece_img.height * scale))
    piece_img = piece_img.resize(new_size, Image.LANCZOS)
    enhancer = ImageEnhance.Brightness(piece_img)
    piece_img = enhancer.enhance(random.uniform(0.6, 1.4))
    enhancer = ImageEnhance.Contrast(piece_img)
    piece_img = enhancer.enhance(random.uniform(0.6, 1.4))
    enhancer = ImageEnhance.Color(piece_img)
    piece_img = enhancer.enhance(random.uniform(0.7, 1.3))
    return add_noise(piece_img) if random.random() < 0.3 else piece_img


def add_noise(img, intensity=0.02):
    img_array = np.array(img)
    noise = np.random.normal(0, intensity * 255, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)


def add_environmental_effects(board):
    if random.random() < 0.2:
        board = board.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if random.random() < 0.3:
        enhancer = ImageEnhance.Color(board)
        board = enhancer.enhance(random.uniform(0.8, 1.2))
    return board


def paste_piece_on_board(board, piece_img, pos):
    x = int(pos[0] - piece_img.width // 2)
    y = int(pos[1] - piece_img.height // 2)
    x = max(0, min(x, board.width - piece_img.width))
    y = max(0, min(y, board.height - piece_img.height))
    if piece_img.mode == 'RGBA':
        mask = piece_img.split()[-1]
        board.paste(piece_img, (x, y), mask)
    else:
        board.paste(piece_img, (x, y))
    return board


def generate_realistic_image(index, split, positions, scenarios):
    board = Image.open(BOARD_IMG).convert("RGBA")
    board_w, board_h = board.size
    print(f"Board size: {board_w}x{board_h}")
    print(f"Number of positions: {len(positions)}")
    scenario = random.choice(scenarios)
    num_pieces = random.randint(scenario["min_pieces"], scenario["max_pieces"])
    available_pieces = scenario["pieces"].copy()
    chosen_pieces = random.sample(available_pieces, min(num_pieces, len(available_pieces)))
    chosen_positions = random.sample(positions, len(chosen_pieces))
    label_lines = []
    for piece_name, pos in zip(chosen_pieces, chosen_positions):
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
                board = paste_piece_on_board(board, piece_img, pos)
                x_center = pos[0] / board_w
                y_center = pos[1] / board_h
                w = piece_img.width / board_w
                h = piece_img.height / board_h
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    board = add_environmental_effects(board)
    img_path = f"{OUTPUT_DIR}/{split}/images/{index}.jpg"
    board.convert("RGB").save(img_path, quality=95)
    label_path = f"{OUTPUT_DIR}/{split}/labels/{index}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))
    print(f"Saved image {index} with {len(label_lines)} pieces")


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    make_dirs()

    temp_board = Image.open(BOARD_IMG)
    positions = get_board_positions(temp_board)
    scenarios = create_game_scenarios()
    print(f"Board dimensions: {temp_board.width}x{temp_board.height}")
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
    with open(f"{OUTPUT_DIR}/classes.txt", "w") as f:
        f.write("\n".join(CLASS_NAMES.values()))
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
