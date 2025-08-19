import argparse
import os
from pathlib import Path
import shutil

import albumentations as A
import cv2


root_dir = Path(__file__).parent.parent
base_dir = root_dir / "scripts" / "augmented"
INPUT_DIR = base_dir / "input"
OUTPUT_DIR = base_dir / "output"
AUGMENTATIONS_PER_IMAGE = 10


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.OneOf([
            A.MotionBlur(p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), p=0.5),
    ], 
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)


def clean_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR / "images")
    os.makedirs(OUTPUT_DIR / "labels")


def save_augmented_image(image, bboxes, class_labels, new_basename):
    new_img_path = OUTPUT_DIR / "images" / f"{new_basename}.jpg"
    new_label_path = OUTPUT_DIR / "labels" / f"{new_basename}.txt"
    cv2.imwrite(new_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    with open(new_label_path, 'w') as f:
        for j, bbox in enumerate(bboxes):
            class_id = class_labels[j]
            x_center, y_center, w, h = bbox
            f.write(f"{class_id:.0f} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Chess AI Training and Detection Tool')
    parser.add_argument('size', type=int, default=AUGMENTATIONS_PER_IMAGE, help='Size of the image')
    args = parser.parse_args()
    size = args.size

    print(f"Starting to augment data with {size } augmentations from '{INPUT_DIR}' to '{OUTPUT_DIR}'...")
    
    # clean output
    clean_output()

    # read images and labels
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_filename in image_files:
        basename, _ = os.path.splitext(img_filename)
        img_path = os.path.join(INPUT_DIR, img_filename)
        label_path = os.path.join(INPUT_DIR, f"{basename}.txt")
        if not os.path.exists(label_path):
            print(f"Warning: Skipping '{img_filename}' because there is no label file.")
            continue
        
        # read image and labels
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                bboxes.append([x_center, y_center, w, h])
                class_labels.append(int(class_id))

        # augment images
        for i in range(size):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']
                if transformed_bboxes:
                    save_augmented_image(
                        transformed_image,
                        transformed_bboxes,
                        transformed_class_labels,
                        f"{basename}_aug_{i}"
                    )
            except Exception as e:
                print(f"Error augmenting image {img_filename}: {e}")
        print(f"Created {size} augmentations for '{img_filename}'")
    print("\n--- COMPLETED! ---")
    print(f"All augmented images saved in '{OUTPUT_DIR}'")


if __name__ == '__main__':
    main()
