"""Auto-label dataset saver for the retrain loop.

Every /detect upload can be saved as a YOLO-format training sample, labeled
with the model's own predictions (pre-annotation). Samples land in per-ISO-week
folders on disk:

    <root>/
      seen_hashes.txt          # global dedupe — one sha1 prefix per line
      2026-W24/
        data.yaml              # YOLO descriptor (18 class names)
        images/<hash>.jpg      # re-encoded pixels (EXIF stripped, matches labels)
        labels/<hash>.txt      # class_id cx cy w h (normalized, 18 classes)
        meta.jsonl             # one line per sample: detected/fen/confidence/...

Labels use the post-NMS piece list (one box per physical piece — duplicate
boxes would poison training) plus every detected landmark. Review happens
locally: scripts/weekly_ingest.py builds per-class galleries from these
folders, scripts/apply_review.py applies the corrections.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np

from .settings import ITEM_CLASS_NAMES

logger = logging.getLogger("ocr_service.dataset")

# save_sample runs in FastAPI's background threadpool; the lock serializes
# seen_hashes.txt reads/appends and week-folder creation.
_lock = threading.Lock()
_seen: Optional[Set[str]] = None


def _load_seen(root: Path) -> Set[str]:
    global _seen
    if _seen is None:
        f = root / "seen_hashes.txt"
        _seen = set(f.read_text(encoding="utf-8").split()) if f.exists() else set()
    return _seen


def _week_dir(root: Path) -> Path:
    d = root / time.strftime("%G-W%V")
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)
    data_yaml = d / "data.yaml"
    if not data_yaml.exists():
        names = ", ".join(f"'{n}'" for n in ITEM_CLASS_NAMES)
        data_yaml.write_text(
            f"nc: {len(ITEM_CLASS_NAMES)}\nnames: [{names}]\n", encoding="utf-8"
        )
    return d


def _yolo_lines(result, width: int, height: int) -> List[str]:
    items = list(result.pieces)
    if result.item_result is not None:
        for lms in result.item_result.landmarks_by_class.values():
            items.extend(lms)
    lines = []
    for it in items:
        x1, y1, x2, y2 = it.bbox
        cx = min(max((x1 + x2) / 2 / width, 0.0), 1.0)
        cy = min(max((y1 + y2) / 2 / height, 0.0), 1.0)
        bw = min(max((x2 - x1) / width, 0.0), 1.0)
        bh = min(max((y2 - y1) / height, 0.0), 1.0)
        lines.append(f"{it.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def save_sample(
    img: np.ndarray, result, root: Path, meta: dict
) -> Optional[str]:
    """Save image + predicted YOLO labels into the current ISO-week folder.

    Never raises — /detect must not fail over dataset bookkeeping.
    Returns the sample hash, or None when skipped (duplicate) or errored.
    """
    try:
        with _lock:
            digest = hashlib.sha1(img.tobytes()).hexdigest()[:16]
            seen = _load_seen(root)
            if digest in seen:
                logger.info("dataset skip duplicate %s", digest)
                return None

            week = _week_dir(root)
            height, width = img.shape[:2]
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                logger.warning("dataset jpg encode failed, sample dropped")
                return None
            lines = _yolo_lines(result, width, height)

            (week / "images" / f"{digest}.jpg").write_bytes(buf.tobytes())
            # An empty .txt is a valid YOLO "null annotation" (no objects).
            (week / "labels" / f"{digest}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            with (week / "meta.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"hash": digest, **meta}, ensure_ascii=False) + "\n")
            with (root / "seen_hashes.txt").open("a", encoding="utf-8") as f:
                f.write(digest + "\n")
            seen.add(digest)

            logger.info(
                "dataset saved %s (%d labels) -> %s", digest, len(lines), week.name
            )

        return digest
    except Exception:
        logger.exception("dataset save failed")
        return None
