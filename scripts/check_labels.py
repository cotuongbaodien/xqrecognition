import os
from glob import glob
from pathlib import Path

# Đường dẫn tới dataset của bạn
base_dir = Path(__file__).parent.parent

images_dir = base_dir / "dataset" / "train" / "images"
labels_dir = base_dir / "dataset" / "train" / "labels"

# Lấy danh sách file (không extension)
images = {os.path.splitext(os.path.basename(f))[0] for f in glob(f"{images_dir}/*")}
labels = {os.path.splitext(os.path.basename(f))[0] for f in glob(f"{labels_dir}/*")}

# Kiểm tra ảnh không có label
missing_labels = images - labels
if missing_labels:
    print("❌ Ảnh bị thiếu file label:", missing_labels)

# Kiểm tra label không có ảnh
missing_images = labels - images
if missing_images:
    print("❌ Label không có ảnh tương ứng:", missing_images)

# Kiểm tra nội dung file label
for label_file in glob(f"{labels_dir}/*.txt"):
    with open(label_file, "r") as f:
        lines = f.readlines()

    if not lines:
        print(f"⚠️  File label rỗng: {label_file}")
        continue

    for i, line in enumerate(lines, 1):
        parts = line.strip().split()
        
        # Phải có đúng 5 phần tử: class, x_center, y_center, width, height
        if len(parts) != 5:
            print(f"❌ {label_file} (dòng {i}) sai số lượng giá trị: {parts}")
            continue

        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            # Giá trị phải nằm trong khoảng 0-1
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print(f"❌ {label_file} (dòng {i}) giá trị out-of-range: {parts}")

        except ValueError:
            print(f"❌ {label_file} (dòng {i}) chứa ký tự không hợp lệ: {parts}")

print("✅ Kiểm tra xong!")