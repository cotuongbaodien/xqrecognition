"""FastAPI server cho boarddetection.

Endpoints:
  GET  /health   → {"status":"ok","model":"loaded"}
  POST /detect   → multipart image → {fen, confidence, pieces_count, errors, image_shape}

Run:
  cd boarddetection
  uvicorn server:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import sys
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

# uvicorn's CLI reconfigures logging after import (root → WARNING, no
# handlers), so basicConfig() here is a no-op. Attach our own stdout handler
# directly to the "ocr_service" logger (children like "ocr_service.dataset"
# inherit it) so detect/dataset lines reach `docker logs`. propagate=False
# avoids double-printing if uvicorn later adds a root handler.
logger = logging.getLogger("ocr_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(_handler)
    logger.propagate = False

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


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# Expected scope claim trong token (Flow B). Portal ký scope="detect" cho endpoint
# này; token cấp cho mục đích khác (nếu sau này có) sẽ không dùng lại được ở đây.
OCR_TOKEN_SCOPE = "detect"

# Chống replay: nhớ các jti đã dùng cho tới khi token hết hạn (single-use).
# Đây là cache IN-PROCESS — đủ vì stack chỉ chạy 1 worker uvicorn (xem
# docker-compose: không có --workers). Nếu scale ra nhiều worker/replica thì
# PHẢI chuyển sang Redis EX=<ttl> để dedup dùng chung. jti -> exp (unix seconds).
_seen_jti: dict[str, float] = {}


def _verify_token(token: str) -> bool:
    """Verify a short-lived HMAC token issued by the portal (Flow B).

    Token = base64url(payload_json) + "." + base64url(HMAC_SHA256(secret, payload_b64)).
    payload = {"exp": <unix_seconds>, "jti": <uuid>, "scope": "detect", ...}. Signed
    with OCR_SHARED_SECRET (shared portal<->ocr; the master secret never leaves the
    servers). Lets the mobile app upload straight to ocr.abcxq.app with a token that
    expires, instead of relaying the image through the portal VPS.

    Checks, in order: chữ ký HMAC → scope → hết hạn → single-use (chống replay).
    Flow A (header X-OCR-Secret tĩnh) KHÔNG đi qua đây nên không bị ảnh hưởng.
    """
    if not token or not OCR_SHARED_SECRET or "." not in token:
        return False
    try:
        payload_b64, sig_b64 = token.rsplit(".", 1)
        expected = hmac.new(
            OCR_SHARED_SECRET.encode(), payload_b64.encode(), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _b64d(sig_b64)):  # 1) chữ ký
            return False
        payload = json.loads(_b64d(payload_b64))
        if payload.get("scope") != OCR_TOKEN_SCOPE:            # 2) scope
            return False
        exp = float(payload.get("exp", 0))
        if exp < time.time():                                  # 3) hết hạn (TTL ~120s)
            return False
        jti = payload.get("jti")
        if not jti:                                            # 4) bắt buộc có jti để single-use
            return False
        now = time.time()
        # Dọn các jti đã hết hạn để cache không phình (window chỉ ~120s).
        if _seen_jti:
            for stale in [k for k, v in _seen_jti.items() if v < now]:
                del _seen_jti[stale]
        if jti in _seen_jti:                                   # 5) đã dùng → replay
            return False
        _seen_jti[jti] = exp
        return True
    except Exception:
        return False


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
    x_ocr_token: str = Header(default=""),
) -> JSONResponse:
    # Auth: long-lived shared secret (portal relay = Flow A) OR a short-lived
    # signed token (mobile uploads straight here = Flow B). Either one passes.
    # auth_via được log ở dòng detect bên dưới để biết request đi đường nào.
    auth_via = "none"
    if OCR_SHARED_SECRET:
        if hmac.compare_digest(x_ocr_secret, OCR_SHARED_SECRET):
            auth_via = "secret"        # Flow A (relay qua portal)
        elif _verify_token(x_ocr_token):
            auth_via = "token"         # Flow B (app upload thẳng)
        else:
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
        "detect auth=%s detected=%s conf=%.3f pieces=%d ms=%d errors=%s",
        auth_via, detected, result.confidence, len(result.pieces), elapsed_ms, errors,
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
