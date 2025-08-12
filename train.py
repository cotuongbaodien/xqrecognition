import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path

# Kiểm tra GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, using CPU")

base_dir = Path(__file__).parent
target_dir = f"{base_dir}/target"
# dataset_dir = "xiangqi_dataset"
# os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
# os.makedirs(f"{dataset_dir}/images/valid", exist_ok=True)
# os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
# os.makedirs(f"{dataset_dir}/labels/valid", exist_ok=True)
CLASS_NAMES = {
    0: "Black Advisor",
    1: "Red Advisor",
    2: "Black Cannon",
    3: "Red Cannon",
    4: "Black Elephant",
    5: "Red Elephant",
    6: "Black King",
    7: "Red King",
    8: "Black Horse",
    9: "Red Horse",
    10: "Black Soldier",
    11: "Red Soldier",
    12: "Black Chariot",
    13: "Red Chariot",
}

DISPLAY_NAMES = {
    "Black Advisor": "A_B",
    "Red Advisor": "A_R",
    "Black Cannon": "Ca_B",
    "Red Cannon": "Ca_R",
    "Black Elephant": "E_B",
    "Red Elephant": "E_R",
    "Black King": "K_B",
    "Red King": "K_R",
    "Black Horse": "H_B",
    "Red Horse": "H_R",
    "Black Soldier": "S_B",
    "Red Soldier": "S_R",
    "Black Chariot": "C_B",
    "Red Chariot": "C_R",
}


def train_model():
    version = int(open(f"{target_dir}/version", "r").read())
    model_size = f"{target_dir}/best.v{version}.pt"
    model = YOLO(model_size)
    print(f"Model loaded: {model_size}")
    print(f"Model summary:")
    model.info()
    training_config = {
        'data': 'dataset/data.yaml',
        'epochs': 50,
        'imgsz': 13440,
        'batch': 16,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': False,
        'device': 0 if torch.cuda.is_available() else 'mps',
        'workers': 8,
        'project': 'runs/detect',
        'name': 'xiangqi_model',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
    }
    model.train(**training_config)
    version += 1
    model.save(f"{target_dir}/best.v{version}.pt")
    with open(f"{target_dir}/version", "w") as f:
        f.write(str(version))
    return model


def detect_pieces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh!")
        return
    version = int(open(f"{target_dir}/version", "r").read())
    model_size = f"{target_dir}/best.v{version}.pt"
    model = YOLO(model_size)
    results = model(image)
    print(f"Model size: {model_size}")
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())
        class_name = CLASS_NAMES.get(class_id, f"Unknown-{class_id}")
        color_name = class_name.split(" ")[0]
        color = (0, 0, 255) if color_name == "Red" else (255, 0, 0)
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )
        label = f"{DISPLAY_NAMES[class_name]}"
        (label_width, label_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )
        cv2.rectangle(
            image,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
            -1
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    cv2.imwrite("output/result.jpg", image)


if __name__ == "__main__":
    train_model()
    # detect_pieces("input/2.png")
