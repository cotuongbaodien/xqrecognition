# Labeling tools
How to label the dataset for YOLOv12?

## 1. LabelImg
- Written in Python + Qt.
- Very lightweight, easy to use.
- Export directly to YOLO format or PascalVOC.
- Use keyboard shortcuts to draw bounding boxes quickly.


Install:
```bash
pip install labelImg
labelImg
```

## 2. Label Studio
- Web app, many users.
- Support teamwork (many people annotate together).
- Can run locally or deploy server.
- Support auto-label plugin (combine with YOLO, HuggingFace model…).

Install:
```bash
pip install label-studio
label-studio
```

## 3. Roboflow Annotate (miễn phí cho dataset nhỏ)
- Web tool, many smart features (auto-annotate, polygon).
- Integrate convert format to YOLOv5/YOLOv8/YOLOv12.
- Support cloud save dataset.

## 4. CVAT (Computer Vision Annotation Tool)
- Powerful, developed by Intel.
- Support video labeling, semi-auto label.
- Integrate AI auto-label (run pretrained model to suggest box).

## 5. Makesense.ai
- Free, run directly on browser.
- No need to install.
- Export directly to YOLO format.


## 6. Recommendation (case training YOLOv12 to detect chess pieces):
- If dataset is small (<1k images) → LabelImg for quick labeling.
- If dataset is large, need teamwork → Label Studio or CVAT.
- If want auto-label + cloud → Roboflow Annotate.
