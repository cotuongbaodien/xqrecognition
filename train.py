import os
import cv2
from ultralytics import YOLO
import torch
from pathlib import Path
import argparse

import yaml

# Kiểm tra GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, using CPU")

if torch.backends.mps.is_available():
    print("Using MPS")
    torch.mps.empty_cache()
else:
    print("No MPS detected, using CPU")


base_dir = Path(__file__).parent
target_dir = f"{base_dir}/target"
CLASS_NAMES = {}
with open(f"{base_dir}/dataset/data.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    CLASS_NAMES.update({i: name for i, name in enumerate(data["names"])})

DISPLAY_NAMES = {
    "Black Advisor": "S",
    "Red Advisor": "S",
    "Black Cannon": "P",
    "Red Cannon": "P",
    "Black Elephant": "T",
    "Red Elephant": "T",
    "Black General": "V",
    "Red General": "V",
    "Black Horse": "M",
    "Red Horse": "M",
    "Black Soldier": "C",
    "Red Soldier": "C",
    "Black Chariot": "X",
    "Red Chariot": "X",
    "intersection": "R"
}


def get_version():
    try:
        version = int(open(f"{target_dir}/version", "r").read())
        return version
    except:
        return 0


def train_model(resume=False):
    version = get_version()
    model_size = f"{target_dir}/best.v{version}.pt" if version > 0 else f"{target_dir}/yolov12s.pt"
    print(f"===============Model size: {model_size}")
    model = YOLO(model_size)
    model.info()
    
    # Cải thiện cấu hình training cho YOLOv12
    training_config = {
        'data': 'dataset/data.yaml',
        'epochs': 200,
        'imgsz': 640,
        'batch': 16,
        'patience': 25,
        'save': True,
        'save_period': 10,
        'cache': 'ram',
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'workers': 6,
        'project': 'runs/detect',
        'name': 'xiangqi_yolo12_model',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': resume,
        'amp': True,  # Bật mixed precision
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.1,  # Tăng dropout
        'val': True,
        # Thêm các hyperparameters mới cho YOLOv12
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
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
    version = get_version()
    model_size = f"{target_dir}/best.v{version}.pt" if version > 0 else f"{target_dir}/yolov12s.pt"
    print(f"===============Model: {model_size}")
    model = YOLO(model_size)
    results = model(image)
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())
        class_name = CLASS_NAMES.get(class_id, f"Unknown-{class_id}")
        if class_name == "intersection":
            continue
        print(f"Class Name: {class_name}, Confidence: {confidence}")
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
    cv2.imwrite(f"output/{image_path.split('/')[-1]}_result.jpg", image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Chess AI Training and Detection Tool')
    parser.add_argument('mode', choices=['train', 'detect'],
                        help='Mode to run: train or detect')
    parser.add_argument('--image', '-i', type=str, default='',
                        help='Image path for detection mode (required when mode=detect)')
    parser.add_argument('--output', '-o', type=str, default='output/result.jpg',
                        help='Output path for detection results (default: output/result.jpg)')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume training from the last checkpoint')

    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting training mode...")
        train_model(args.resume)
        print("Training completed!")

    elif args.mode == 'detect':
        if args.image and os.path.exists(args.image):
            print(f"Starting detection mode on image: {args.image}")
            detect_pieces(args.image)
            print(f"Detection completed! Result saved to: {args.output}")
        else:
            INPUT_DIR = base_dir / "input"
            image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_filename in image_files:
                img_path = INPUT_DIR / img_filename
                print(f"Starting detection mode on image: {img_path}")
                detect_pieces(img_path.as_posix())
                print(f"Detection completed! Result saved to: {args.output if args.output else 'output'}")
