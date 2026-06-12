"""One-off backfill: auto-label a folder of historical images into the dataset.

Usage (inside the ocr container):
    python -m boarddetection.backfill /tmp/ocr_backfill [batch-name]

For every image in the folder:
  - run the recognizer and write image+YOLO labels into
    dataset/<batch-name>/{images,labels}/ (+ data.yaml, meta.jsonl),
  - append the pixel hash to dataset/seen_hashes.txt so the weekly loop
    never re-saves these images — from now on weekly folders hold only
    NEW user uploads,
  - if ROBOFLOW_API_KEY/ROBOFLOW_PROJECT are set, upload to the Roboflow
    Annotate queue under the same batch name (review there, then train).

Re-runnable: already-seen hashes are skipped, so a crashed run just resumes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

from . import roboflow_uploader
from .dataset_saver import _load_seen, _yolo_lines, _lock
from .pipeline import XiangqiRecognizer
from .settings import ITEM_CLASS_NAMES

DATASET_ROOT = Path(__file__).parent / "dataset"


def main() -> None:
    src = Path(sys.argv[1])
    batch = sys.argv[2] if len(sys.argv) > 2 else "backfill-" + time.strftime("%Y-%m-%d")
    out = DATASET_ROOT / batch
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    data_yaml = out / "data.yaml"
    if not data_yaml.exists():
        names = ", ".join(f"'{n}'" for n in ITEM_CLASS_NAMES)
        data_yaml.write_text(
            f"nc: {len(ITEM_CLASS_NAMES)}\nnames: [{names}]\n", encoding="utf-8"
        )

    recognizer = XiangqiRecognizer()
    seen = _load_seen(DATASET_ROOT)
    files = sorted(src.glob("*.jpg"))
    stats = {"saved": 0, "dup": 0, "decode_fail": 0, "uploaded": 0, "upload_fail": 0}

    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            stats["decode_fail"] += 1
            continue
        import hashlib
        digest = hashlib.sha1(img.tobytes()).hexdigest()[:16]
        if digest in seen:
            stats["dup"] += 1
            continue

        result = recognizer.recognize_image(img)
        height, width = img.shape[:2]
        lines = _yolo_lines(result, width, height)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            stats["decode_fail"] += 1
            continue

        (out / "images" / f"{digest}.jpg").write_bytes(buf.tobytes())
        (out / "labels" / f"{digest}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        with (out / "meta.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "hash": digest,
                "source": f.name,
                "fen": result.fen,
                "confidence": round(float(result.confidence), 3),
                "pieces_count": len(result.pieces),
                "errors": result.errors or [],
            }, ensure_ascii=False) + "\n")
        with _lock:
            with (DATASET_ROOT / "seen_hashes.txt").open("a", encoding="utf-8") as fh:
                fh.write(digest + "\n")
            seen.add(digest)
        stats["saved"] += 1

        if roboflow_uploader.enabled():
            image_id = roboflow_uploader.upload_sample(
                buf.tobytes(), digest, lines, batch=batch
            )
            if image_id:
                stats["uploaded"] += 1
                with (DATASET_ROOT / "uploaded.txt").open("a", encoding="utf-8") as fh:
                    fh.write(f"{digest} {image_id}\n")
            else:
                stats["upload_fail"] += 1

        if i % 25 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] {stats}", flush=True)

    print("DONE", json.dumps(stats))


if __name__ == "__main__":
    main()
