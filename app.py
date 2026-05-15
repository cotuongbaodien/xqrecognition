"""
FastAPI web service for Xiangqi Recognition System.
Provides REST API for detecting chess pieces and generating FEN notation.
"""

import io
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ITEMS_MODEL
from src.pipeline import XiangqiRecognizer


# Initialize FastAPI app
app = FastAPI(
    title="Xiangqi Recognition API",
    description="API for detecting Xiangqi (Chinese Chess) pieces and generating FEN notation",
    version="1.0.0",
)

# Global recognizer instance
recognizer: Optional[XiangqiRecognizer] = None


class DetectionResponse(BaseModel):
    """Response model for detection endpoint."""
    fen: str
    pieces: list
    piece_count: int
    confidence: float
    errors: list


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    items_model_loaded: bool


def get_recognizer() -> XiangqiRecognizer:
    """Get or initialize the recognizer instance."""
    global recognizer
    if recognizer is None:
        recognizer = XiangqiRecognizer(items_model_path=str(ITEMS_MODEL))
    return recognizer


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    print("Initializing Xiangqi Recognition System...")
    try:
        get_recognizer()
        print("System initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize recognizer: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Xiangqi Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - Detect pieces and generate FEN from image",
            "/detect/visualize": "POST - Detect pieces and return visualization",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    rec = get_recognizer()
    return HealthResponse(
        status="healthy",
        items_model_loaded=rec.item_detector.model is not None,
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    """
    Detect chess pieces and generate FEN notation from an image.

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        DetectionResponse with FEN string and detected pieces.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )

    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Run recognition
        rec = get_recognizer()
        result = rec.recognize_image(image)

        return DetectionResponse(
            fen=result.fen,
            pieces=[p.to_dict() for p in result.pieces],
            piece_count=len(result.pieces),
            confidence=result.confidence,
            errors=result.errors,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/visualize")
async def detect_visualize(file: UploadFile = File(...)):
    """
    Detect chess pieces and return visualization image.

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        PNG image with detection visualization.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )

    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Run recognition with visualization
        rec = get_recognizer()
        result = rec.recognize_image(image, visualize=True)

        if result.visualization is None:
            raise HTTPException(status_code=500, detail="Could not create visualization")

        # Encode visualization as PNG
        _, encoded = cv2.imencode(".png", result.visualization)
        return StreamingResponse(
            io.BytesIO(encoded.tobytes()),
            media_type="image/png",
            headers={
                "X-FEN": result.fen,
                "X-Piece-Count": str(len(result.pieces)),
                "X-Confidence": f"{result.confidence:.4f}",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/json-with-image")
async def detect_json_with_image(file: UploadFile = File(...)):
    """
    Detect chess pieces and return both JSON data and base64-encoded visualization.

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        JSON with detection results and base64 visualization image.
    """
    import base64

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )

    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Run recognition with visualization
        rec = get_recognizer()
        result = rec.recognize_image(image, visualize=True)

        # Encode visualization as base64
        visualization_b64 = None
        if result.visualization is not None:
            _, encoded = cv2.imencode(".png", result.visualization)
            visualization_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        return JSONResponse({
            "fen": result.fen,
            "pieces": [p.to_dict() for p in result.pieces],
            "piece_count": len(result.pieces),
            "confidence": result.confidence,
            "errors": result.errors,
            "visualization": visualization_b64,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Xiangqi Recognition API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
