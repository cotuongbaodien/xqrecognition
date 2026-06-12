"""Best-effort Roboflow uploader for auto-labeled samples.

Pushes each saved sample (image + YOLO prediction labels) straight to
Roboflow via REST. Images land in the **Annotate tab** under a weekly batch
(`auto-<ISO week>`) — NOT in the main dataset. The reviewer opens the batch,
fixes labels, approves ("Add to Dataset"), then trains a new version.

Config (env):
    ROBOFLOW_API_KEY   private API key — empty = uploader disabled
    ROBOFLOW_PROJECT   project slug (the {project} in app.roboflow.com URLs)

Endpoints (docs.roboflow.com → REST API → Manage Images):
    POST api.roboflow.com/dataset/{project}/upload?api_key&name&batch
         multipart field "file"; response {"id", "success", "duplicate"}
    POST api.roboflow.com/dataset/{project}/annotate/{id}?api_key&name=x.txt
         JSON {"annotationFile": <yolo txt>, "labelmap": {"0": "...", ...}}

Failures only log — the local weekly dataset folder is the source of truth,
so anything that fails to upload can still be uploaded by hand.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
import uuid
from typing import Dict, List, Optional

from .settings import ITEM_CLASS_NAMES

logger = logging.getLogger("ocr_service.roboflow")

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_PROJECT = os.environ.get("ROBOFLOW_PROJECT", "")
_API = "https://api.roboflow.com"
_TIMEOUT = 20

LABELMAP: Dict[str, str] = {str(i): n for i, n in enumerate(ITEM_CLASS_NAMES)}


def enabled() -> bool:
    return bool(ROBOFLOW_API_KEY and ROBOFLOW_PROJECT)


def _post(url: str, body: bytes, content_type: str) -> dict:
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload_sample(
    jpg_bytes: bytes, digest: str, yolo_lines: List[str], batch: str
) -> Optional[str]:
    """Upload one image + its prediction labels into the weekly Annotate batch.

    Returns the Roboflow image id, or None on failure (logged, never raises).
    """
    if not enabled():
        return None
    try:
        # 1) Image → Annotate tab, batch "auto-<week>" (review queue, not dataset)
        query = urllib.parse.urlencode({
            "api_key": ROBOFLOW_API_KEY,
            "name": f"{digest}.jpg",
            "batch": batch,
        })
        boundary = uuid.uuid4().hex
        body = (
            (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="file"; filename="{digest}.jpg"\r\n'
             f"Content-Type: image/jpeg\r\n\r\n").encode()
            + jpg_bytes
            + f"\r\n--{boundary}--\r\n".encode()
        )
        up = _post(
            f"{_API}/dataset/{ROBOFLOW_PROJECT}/upload?{query}",
            body, f"multipart/form-data; boundary={boundary}",
        )
        image_id = up.get("id")
        if not image_id:
            logger.warning("roboflow upload rejected %s: %s", digest, up)
            return None
        if up.get("duplicate"):
            logger.info("roboflow duplicate %s (id=%s)", digest, image_id)

        # 2) Prediction labels (YOLO txt + labelmap) attached to the image
        if yolo_lines:
            query = urllib.parse.urlencode({
                "api_key": ROBOFLOW_API_KEY,
                "name": f"{digest}.txt",
            })
            ann = _post(
                f"{_API}/dataset/{ROBOFLOW_PROJECT}/annotate/{image_id}?{query}",
                json.dumps({
                    "annotationFile": "\n".join(yolo_lines),
                    "labelmap": LABELMAP,
                }).encode("utf-8"),
                "application/json",
            )
            if not (ann.get("success") or ann.get("id")):
                logger.warning("roboflow annotate failed %s: %s", digest, ann)

        logger.info("roboflow uploaded %s -> batch %s (id=%s)", digest, batch, image_id)
        return image_id
    except Exception:
        logger.exception("roboflow upload failed %s", digest)
        return None
