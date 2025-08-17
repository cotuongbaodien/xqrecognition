import os
import cv2
import yaml
from pathlib import Path

# ==== CONFIG ====
base_dir = Path(__file__).parent.parent
DATASET_DIR = f"{base_dir}/dataset"
SPLIT = "train"   # "train" hoặc "val"

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
with open(f"{base_dir}/dataset/data.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    CLASS_NAMES.update({i: name for i, name in enumerate(data["names"])})
# =================


def visualize(split="train"):
    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")
    out_dir = os.path.join(DATASET_DIR, split, "preview")
    os.makedirs(out_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.rsplit(".", 1)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Không đọc được ảnh: {img_name}")
            continue
        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, bw, bh = parts
                    cls = int(cls)
                    x, y, bw, bh = map(float, (x, y, bw, bh))

                    # Convert YOLO format → pixel
                    x1 = int((x - bw/2) * w)
                    y1 = int((y - bh/2) * h)
                    x2 = int((x + bw/2) * w)
                    y2 = int((y + bh/2) * h)

                    color = (0, 255, 0)  # xanh lá
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
                    label = DISPLAY_NAMES.get(label, label)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, img)
        print(f"[OK] {out_path} saved.")


if __name__ == "__main__":
    visualize(SPLIT)
    print("✅ Preview saved in dataset/preview/")
