#!/usr/bin/env python3
import os
import random
from PIL import Image, ImageEnhance
import shutil

BOARD_IMG = "scripts/ban_co.png"
PIECES_DIR = "scripts/pieces"
OUTPUT_DIR = "scripts/dataset"
NUM_TRAIN = 5000
NUM_VAL = 1000
CLASSES = [
    "Advisor_black",
    "Advisor_red",
    "Cannon_black",
    "Cannon_red",
    "Elephant_black",
    "Elephant_red",
    "King_black",
    "King_red",
    "Horse_black",
    "Horse_red",
    "Soldier_black",
    "Soldier_red",
    "Chariot_black",
    "Chariot_red",
]
ROWS = 10
COLS = 9


def make_dirs():
    for split in ["train", "valid"]:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)


def get_board_positions(board_w, board_h):
    cell_w = board_w / (COLS - 1)
    cell_h = board_h / (ROWS - 1)
    positions = []
    for r in range(ROWS):
        for c in range(COLS):
            x = c * cell_w
            y = r * cell_h
            positions.append((x, y))
    return positions


def random_augment(piece_img):
    angle = random.uniform(-10, 10)
    piece_img = piece_img.rotate(angle, expand=True)
    scale = random.uniform(0.9, 1.1)
    new_size = (int(piece_img.width * scale), int(piece_img.height * scale))
    piece_img = piece_img.resize(new_size, Image.LANCZOS)
    enhancer = ImageEnhance.Brightness(piece_img)
    piece_img = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(piece_img)
    piece_img = enhancer.enhance(random.uniform(0.8, 1.2))
    return piece_img


def make_label(pos, board_w, board_h):
    class_id = random.randint(0, len(CLASSES)-1)
    piece_file = f"{PIECES_DIR}/{CLASSES[class_id]}.png"
    piece_img = Image.open(piece_file).convert("RGBA")
    piece_img = random_augment(piece_img)
    x_center = (pos[0]) / board_w
    y_center = (pos[1]) / board_h
    w = piece_img.width / board_w
    h = piece_img.height / board_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def generate_one_image(index, split, positions):
    board = Image.open(BOARD_IMG).convert("RGBA")
    board_w, board_h = board.size
    num_pieces = random.randint(3, 15)
    chosen_positions = random.sample(positions, num_pieces)
    label_lines = [make_label(pos, board_w, board_h) for pos in chosen_positions]
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
    for i in range(NUM_TRAIN):
        generate_one_image(i, "train", positions)
    for i in range(NUM_VAL):
        generate_one_image(i, "valid", positions)

    with open(f"{OUTPUT_DIR}/classes.txt", "w") as f:
        f.write("\n".join(CLASSES))

    print("Dataset has been generated!")


if __name__ == "__main__":
    main()
