import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

root_dir = Path(__file__).parent.parent
base_dir = root_dir / "scripts" / "coco"

for target in ["train", "valid", "test"]:
    print(f"Processing {target}...")
    input_dir = base_dir / "input" / target
    coco_json = input_dir / "_annotations.coco.json"
    labels_dir = base_dir / "output" / target / "labels"
    images_dir = base_dir / "output" / target / "images"
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    with open(coco_json, 'r') as f:
        coco = json.load(f)
    image_info = {img["id"]: img for img in coco["images"]}
    cat2id = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)
    for image_id, anns in tqdm(ann_by_img.items()):
        img_file = image_info[image_id]["file_name"]
        img_path = input_dir / img_file
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        w, h = img.size
        label_path = labels_dir / (img_file.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for ann in anns:
                cat_id = ann["category_id"]
                class_id = cat2id[cat_id]
                x, y, bw, bh = ann["bbox"]
                x_c = (x + bw / 2) / w
                y_c = (y + bh / 2) / h
                bw /= w
                bh /= h
                f.write(f"{class_id - 1} {x_c} {y_c} {bw} {bh}\n")
        
        new_img_path = images_dir / img_file
        img.save(new_img_path)
        os.remove(img_path)

print("Finished")
