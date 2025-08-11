# Xiangqi Chess Detection Project

A computer vision project for detecting and recognizing Xiangqi (Chinese Chess) pieces on a game board using YOLOv8 object detection.

## Overview

This project implements an AI system that can:
- Detect the 9x10 grid of a Xiangqi board
- Recognize and classify 14 different types of chess pieces (7 red pieces + 7 black pieces)
- Map detected pieces to their correct board positions
- Generate a complete board state representation

## Features

### Piece Detection
- **Red Pieces (R_)**:
  - R_K: Red King (Tướng đỏ)
  - R_C: Red Chariot (Xe đỏ)
  - R_H: Red Horse (Mã đỏ)
  - R_E: Red Elephant (Tượng đỏ)
  - R_A: Red Advisor (Sĩ đỏ)
  - R_P: Red Cannon (Pháo đỏ)
  - R_S: Red Soldier (Tốt đỏ)

- **Black Pieces (B_)**:
  - B_K: Black King (Tướng đen)
  - B_C: Black Chariot (Xe đen)
  - B_H: Black Horse (Mã đen)
  - B_E: Black Elephant (Tượng đen)
  - B_A: Black Advisor (Sĩ đen)
  - B_P: Black Cannon (Pháo đen)
  - B_S: Black Soldier (Tốt đen)

### Board Grid Detection
- Automatically detects the 9x10 grid structure
- Uses Hough Transform for line detection
- Maps pieces to standard chess notation (a0-i9)

## Project Structure

```
chess_ai/
├── detection.py              # Main detection script
├── xiangqi_dataset/         # Training dataset
│   ├── images/
│   │   ├── train/          # Training images
│   │   └── val/            # Validation images
│   ├── labels/
│   │   ├── train/          # Training labels
│   │   └── val/            # Validation labels
│   └── data.yaml           # Dataset configuration
├── input/                   # Input images for detection
├── yolov8s.pt              # Pre-trained YOLOv8 model
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Dependencies
```bash
pip install ultralytics opencv-python torch numpy
```

## Training

### 1. Prepare Dataset
Organize your training data in the following structure:
```
xiangqi_dataset/
├── images/
│   ├── train/     # Training images
│   └── val/       # Validation images
├── labels/
│   ├── train/     # Training labels (YOLO format)
│   └── val/       # Validation labels (YOLO format)
└── data.yaml      # Dataset configuration
```

### 2. Train the Model
```python
from detection import train_model

# Train the YOLOv8 model
model = train_model()
```

Training parameters:
- **Model**: YOLOv8s (small variant)
- **Image size**: 416x416
- **Epochs**: 50
- **Batch size**: 8
- **Output**: `xiangqi_yolov8.pt`

## Detection

### Usage
```python
from detection import detect_pieces

# Detect pieces on a board image
detect_pieces("input/board.jpg")
```

### Output
The system will:
1. Detect the board grid
2. Recognize all pieces
3. Map pieces to board positions
4. Save annotated image as `output.jpg`
5. Print board state in console

Example output:
```
R_K: e9
B_C: a0
R_P: c7
...
```

This README provides a comprehensive overview of your Xiangqi chess detection project, including:

1. **Project Overview**: Clear description of what the system does
2. **Features**: Detailed list of detected pieces and capabilities
3. **Installation**: Step-by-step setup instructions
4. **Training Guide**: How to prepare data and train the model
5. **Usage Examples**: Practical code examples
6. **Technical Details**: Algorithm explanations and architecture
7. **Performance Information**: Hardware requirements and optimization
8. **Future Roadmap**: Potential improvements

The README is structured to be helpful for both users who want to use the system and developers who want to understand or contribute to the codebase.

## Key Functions

### `detect_board_grid(image)`
- Detects 9x10 grid using Hough Transform
- Returns grid points and position mapping
- Uses chess notation (a0-i9)

### `map_to_board(boxes, grid, board_map)`
- Maps detected pieces to board positions
- Finds closest grid point for each piece
- Returns complete board state

### `train_model()`
- Trains YOLOv8 model on Xiangqi dataset
- Saves trained model as `xiangqi_yolov8.pt`

### `detect_pieces(image_path)`
- Main detection function
- Processes input image and generates results

## Technical Details

### Grid Detection Algorithm
1. Convert image to grayscale
2. Apply histogram equalization
3. Detect edges using Canny algorithm
4. Use Hough Transform for line detection
5. Classify horizontal and vertical lines
6. Extract 9x10 grid intersection points

### Model Architecture
- **Base Model**: YOLOv8s
- **Input Size**: 416x416 pixels
- **Classes**: 14 (7 red + 7 black pieces)
- **Training**: Transfer learning from pre-trained weights

## Performance

### Hardware Requirements
- **Minimum**: CPU-only (slower inference)
- **Recommended**: CUDA GPU for faster training and inference

### GPU Support
The system automatically detects and uses available GPUs:
```python
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected, using CPU")
```

## Usage Examples

### Training a New Model
```python
# Uncomment in detection.py
train_model()
```

### Detecting Pieces
```python
# Place your board image in input/board.jpg
detect_pieces("input/board.jpg")
```

### How to work on this project
Install dependencies
```bash
cd chess_ai
pyenv local 3.10.12
pyenv virtualenv 3.10.12 chess_ai_env
pyenv activate chess_ai_env
pip install -r requirements.txt
```

Run the project
```bash
python detection.py
```

### Result
![Result](result.jpg)

## Future Improvements

1. **Real-time Detection**: Implement video stream processing
2. **Move Validation**: Add rules to validate legal moves
3. **Game State Tracking**: Track complete game history
4. **Multi-board Support**: Handle different board styles
5. **Performance Optimization**: Model quantization and optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- YOLOv8 by Ultralytics for object detection
- OpenCV for computer vision operations
- PyTorch for deep learning framework
