"""FastAPI server cho boarddetection.

Endpoints:
  GET  /health   → {"status":"ok","model":"loaded"}
  POST /detect   → multipart image → {fen, confidence, pieces_count, errors, image_shape}

Run:
  cd boarddetection
  uvicorn server:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .dataset_saver import save_sample
from .pipeline import XiangqiRecognizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ocr_service")

MAX_BYTES = 8 * 1024 * 1024
MIN_CONFIDENCE = float(os.environ.get("OCR_MIN_CONFIDENCE", "0.35"))
MIN_PIECES = int(os.environ.get("OCR_MIN_PIECES", "5"))
# Optional shared-secret. Nếu set, mọi request /detect phải gửi đúng header
# X-OCR-Secret. Bỏ trống = không check (backward-compatible cho localhost).
OCR_SHARED_SECRET = os.environ.get("OCR_SHARED_SECRET", "")
# Auto-label dataset cho vòng retrain Roboflow hàng tuần (dataset_saver.py).
# DIR rỗng = tắt. MODE: "all" lưu mọi ảnh, "failed" chỉ lưu ảnh detected=false.
DATASET_DIR = os.environ.get("OCR_DATASET_DIR", "")
DATASET_MODE = os.environ.get("OCR_DATASET_MODE", "all").lower()

app = FastAPI(title="boarddetection", version="1.0.0")
_recognizer: Optional[XiangqiRecognizer] = None


@app.on_event("startup")
def _load_model() -> None:
    global _recognizer
    t0 = time.time()
    logger.info("Loading XiangqiRecognizer model...")
    _recognizer = XiangqiRecognizer()
    logger.info("Model loaded in %.2fs", time.time() - t0)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "loaded" if _recognizer is not None else "loading",
    }


@app.post("/detect")
async def detect(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    x_ocr_secret: str = Header(default=""),
) -> JSONResponse:
    if OCR_SHARED_SECRET and x_ocr_secret != OCR_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    if _recognizer is None:
        raise HTTPException(status_code=503, detail="model_not_ready")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty_body")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="too_large")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="decode_failed")

    t0 = time.time()
    try:
        result = _recognizer.recognize_image(img, piece_confidence=MIN_CONFIDENCE)
    except Exception as exc:  # noqa: BLE001
        logger.exception("recognize failed")
        raise HTTPException(status_code=500, detail=f"recognize_failed: {exc}") from exc

    elapsed_ms = int((time.time() - t0) * 1000)

    detected = (
        bool(result.fen)
        and result.fen != "9/9/9/9/9/9/9/9/9/9"
        and len(result.pieces) >= MIN_PIECES
    )

    # A FEN missing either general is unplayable — the mobile app's check
    # logic breaks on a kingless board ("in check" on both sides, 2026-06-12).
    # Better to return undetected and let the user reshoot.
    errors = list(result.errors or [])
    if detected:
        board_fen = result.fen.split()[0]
        missing = [
            err
            for sym, err in (("K", "missing_red_general"), ("k", "missing_black_general"))
            if sym not in board_fen
        ]
        if missing:
            errors.extend(missing)
            detected = False

    logger.info(
        "detect detected=%s conf=%.3f pieces=%d ms=%d errors=%s",
        detected, result.confidence, len(result.pieces), elapsed_ms, errors,
    )

    if DATASET_DIR and (DATASET_MODE != "failed" or not detected):
        background_tasks.add_task(
            save_sample, img, result, Path(DATASET_DIR),
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "detected": detected,
                "fen": result.fen if detected else None,
                "confidence": round(float(result.confidence), 3),
                "pieces_count": len(result.pieces),
                "errors": errors,
                "processing_ms": elapsed_ms,
            },
        )

    return JSONResponse(
        {
            "detected": detected,
            "fen": result.fen if detected else None,
            "confidence": round(float(result.confidence), 3),
            "pieces_count": len(result.pieces),
            "image_shape": list(result.image_shape) if result.image_shape else None,
            "errors": errors,
            "processing_ms": elapsed_ms,
        }
    )
