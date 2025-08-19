import argparse
import os
import cv2
import yaml
from pathlib import Path

root_dir = Path(__file__).parent.parent
DATASET_DIR = root_dir / "dataset"
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
CLASS_NAMES = {}
with open(DATASET_DIR / "data.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    CLASS_NAMES.update({i: name for i, name in enumerate(data["names"])})


def visualize(split="train"):
    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"
    out_dir = DATASET_DIR / split / "preview"
    os.makedirs(out_dir, exist_ok=True)

    for img_name in filter(lambda x: x.lower().endswith((".jpg", ".jpeg", ".png")), os.listdir(img_dir)):
        img_path = img_dir / img_name
        lbl_path = lbl_dir / (img_name.rsplit(".", 1)[0] + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Cannot read image: {img_name}")
            continue
        if not lbl_path.exists():
            print(f"[ERROR] Label file not found: {lbl_path}")
            continue
        
        h, w = img.shape[:2]
        with open(lbl_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"[ERROR] Invalid label format: {line}")
                    continue
            
                cls, x, y, bw, bh = parts
                cls = int(cls)
                x, y, bw, bh = map(float, (x, y, bw, bh))
                # Convert YOLO format to pixel
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)
                color = (0, 255, 0)  # green
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
                label = DISPLAY_NAMES.get(label, label)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = out_dir / img_name
        cv2.imwrite(out_path, img)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chess AI Training and Detection Tool')
    parser.add_argument('split', choices=['train', 'valid', 'test'], help='Split to visualize')
    args = parser.parse_args()
    split = args.split
    visualize(split)
    print(f"Preview saved in dataset/{split}/preview/")
