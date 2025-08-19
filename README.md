# Chinese Chess (Xiangqi) AI Detector

A computer vision system that detects and identifies Chinese chess pieces on a board using YOLO (You Only Look Once) object detection. This project includes a FastAPI web service for real-time piece detection and a comprehensive training pipeline.

##  Features

- **Real-time Chinese Chess Piece Detection**: Detects all 14 types of Chinese chess pieces (7 red + 7 black pieces)
- **Board Perspective Correction**: Automatically detects and warps the chess board to a standard perspective
- **Cell Mapping**: Maps detected pieces to their corresponding board positions (9x10 grid)
- **RESTful API**: FastAPI-based web service for easy integration
- **Docker Support**: Containerized deployment ready
- **Comprehensive Training Pipeline**: Complete dataset generation and model training workflow

## ️ Architecture

### Core Components

- **YOLO Model**: Uses Ultralytics YOLO for piece detection
- **Board Detection**: OpenCV-based board contour detection and perspective transformation
- **API Service**: FastAPI server with automatic OpenAPI documentation
- **Training Pipeline**: Custom dataset generation and model training scripts

### Supported Piece Types

| Piece | Red | Black | Display Symbol |
|-------|-----|-------|----------------|
| General | 将 | 帅 | V |
| Advisor | 士 | 仕 | S |
| Elephant | 象 | 相 | T |
| Horse | 马 | 马 | M |
| Chariot | 车 | 车 | X |
| Cannon | 炮 | 炮 | P |
| Soldier | 兵 | 卒 | C |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chess_ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:8000`

### Using Docker

```bash
# Build the Docker image
docker build -t chess-ai .

# Run the container
docker run -p 8000:8000 chess-ai
```

## 📖 API Documentation

### Endpoint: `POST /detect`

Detects Chinese chess pieces in an uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "pieces": [
    {
      "name": "Red General",
      "confidence": 0.95,
      "bbox": [100, 150, 200, 250],
      "center": [150, 200],
      "cell": {
        "col": 4,
        "row": 0,
        "cell_name": "c4_r0"
      }
    }
  ]
}
```

**Response Fields:**
- `name`: Piece type and color
- `confidence`: Detection confidence (0-1)
- `bbox`: Bounding box coordinates [x1, y1, x2, y2]
- `center`: Center point coordinates [x, y]
- `cell`: Board position mapping (9x10 grid)

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

## 🎓 Training Your Own Model

### 1. Dataset Generation

The project includes scripts to generate synthetic training data:

```bash
# Generate synthetic dataset
cd scripts
python gen_dataset.py

# Split dataset into train/validation sets
python split_dataset.py
```

### 2. Model Training

```bash
# Train a new model
python train.py train

# Resume training from checkpoint
python train.py train --resume

# Test detection on an image
python train.py detect --image path/to/image.jpg
```

### Training Configuration

The training script uses YOLOv12 with the following optimized settings:
- **Epochs**: 200
- **Batch Size**: 16
- **Image Size**: 640x640
- **Optimizer**: Auto (AdamW)
- **Learning Rate**: 0.01 with cosine annealing
- **Data Augmentation**: Mosaic, mixup, HSV adjustments

## 📁 Project Structure

```
chess_ai/
├── app.py                 # FastAPI application
├── train.py              # Training and detection script
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── dataset/             # Training dataset
│   ├── data.yaml        # Dataset configuration
│   ├── train/           # Training images and labels
│   ├── valid/           # Validation images and labels
│   └── test/            # Test images and labels
├── scripts/             # Utility scripts
│   ├── gen_dataset.py   # Dataset generation
│   ├── augmented.py     # Data augmentation
│   ├── split_dataset.py # Dataset splitting
│   └── ...
├── target/              # Trained models
├── input/               # Input images for testing
├── output/              # Detection results
└── runs/                # Training logs and outputs
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the object detection framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision operations

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is designed for Chinese Chess (Xiangqi) detection. For Western Chess, you would need to modify the piece classes and board dimensions accordingly.
