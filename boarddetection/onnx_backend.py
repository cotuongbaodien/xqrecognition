"""ONNX Runtime backend — drop-in cho `ultralytics.YOLO` (CPU, không torch).

Mục tiêu: chạy OCR trên VPS CPU bằng onnxruntime, GIỮ NGUYÊN toàn bộ hậu xử lý/FEN.
`OnnxYOLO(path)` được gọi y như model ultralytics:

    results = model(image_bgr, conf=0.25, imgsz=960, verbose=False)   # detect
    result  = model(image_bgr, conf=0.25, imgsz=640, verbose=False)[0] # segment

và trả về object mô phỏng đúng các field code đọc:
  - detect:  result.boxes.cls[i] / .conf[i] / .xyxy[i]  (mỗi cái .cpu().numpy())
  - segment: result.masks.xy (list polygon toạ độ GỐC) + result.boxes.cls

Logic postprocess numpy-hoá từ ultralytics ops (non_max_suppression / process_mask /
scale_boxes / masks2segments) để khớp kết quả torch — verify parity ở test/bench.

Số tự suy ra từ ONNX (task = #outputs, nc = chiều output, imgsz = input shape) nên
không cần cấu hình tay.
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

# Mặc định khớp ultralytics predict: iou=0.7, max_det=300.
_DEFAULT_IOU = 0.7
_MAX_DET = 300
_PAD_VALUE = 114


# --------------------------------------------------------------------------- #
# Shim mô phỏng tensor/boxes/masks/result của ultralytics (chỉ phần code dùng)
# --------------------------------------------------------------------------- #
class _T:
    """Bọc 1 ndarray, mô phỏng `tensor[i].cpu().numpy()` / `.astype()` / `.tolist()`."""
    __slots__ = ("_a",)

    def __init__(self, a):
        self._a = np.asarray(a)

    def __getitem__(self, i):
        return _T(self._a[i])

    def __len__(self):
        return len(self._a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def astype(self, t):
        return _T(self._a.astype(t))

    def tolist(self):
        return self._a.tolist()


class _Boxes:
    def __init__(self, cls: np.ndarray, conf: np.ndarray, xyxy: np.ndarray):
        self.cls = _T(cls)
        self.conf = _T(conf)
        self.xyxy = _T(xyxy)

    def __len__(self):
        return len(self.cls)


class _Masks:
    def __init__(self, xy: List[np.ndarray]):
        self.xy = xy  # list[ ndarray Nx2 float32 ] — toạ độ ảnh GỐC

    def __len__(self):
        return len(self.xy)


class _Result:
    def __init__(self, boxes: _Boxes, masks: Optional[_Masks] = None):
        self.boxes = boxes
        self.masks = masks


# --------------------------------------------------------------------------- #
# Helpers (letterbox / nms / mask)
# --------------------------------------------------------------------------- #
def _letterbox(img: np.ndarray, new: int):
    """Square letterbox giữ tỉ lệ, pad 114 (khớp ultralytics LetterBox center)."""
    h0, w0 = img.shape[:2]
    r = min(new / h0, new / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new - nw) / 2, (new - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(_PAD_VALUE, _PAD_VALUE, _PAD_VALUE),
    )
    return out, r, left, top


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Greedy NMS thuần numpy. Trả index giữ lại (theo thứ tự score giảm dần)."""
    if boxes_xyxy.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# OnnxYOLO
# --------------------------------------------------------------------------- #
class OnnxYOLO:
    def __init__(self, path: str, task: Optional[str] = None):
        so = ort.SessionOptions()
        n = os.environ.get("OCR_ONNX_THREADS")
        if n:
            so.intra_op_num_threads = int(n)
        self.session = ort.InferenceSession(
            path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._inp = self.session.get_inputs()[0].name
        self._outs = [o.name for o in self.session.get_outputs()]
        # task: 2 output (det + proto) => segment, 1 output => detect.
        self.task = task or ("segment" if len(self._outs) >= 2 else "detect")
        shp = self.session.get_inputs()[0].shape  # [1,3,H,W]
        self._imgsz = int(shp[2]) if isinstance(shp[2], int) else 640

    # ultralytics gọi: model(img, conf=, imgsz=, verbose=) -> list[result]
    def __call__(self, image, conf: float = 0.25, imgsz: Optional[int] = None,
                 iou: float = _DEFAULT_IOU, verbose: bool = False, **_):
        sz = self._imgsz  # cố định theo export (imgsz arg chỉ để tương thích chữ ký)
        lb, r, pad_x, pad_y = _letterbox(image, sz)
        blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)
        outputs = self.session.run(None, {self._inp: blob})

        h0, w0 = image.shape[:2]
        if self.task == "segment":
            return [self._post_segment(outputs, conf, iou, r, pad_x, pad_y, w0, h0)]
        return [self._post_detect(outputs, conf, iou, r, pad_x, pad_y, w0, h0)]

    # ----- detection ----- #
    def _decode_dets(self, pred: np.ndarray, conf: float):
        """pred: [4+nc(+nm), N] -> trả (xyxy_lb, score, cls, coeff|None) sau lọc conf."""
        pred = pred.T  # [N, 4+nc(+nm)]
        boxes = pred[:, :4]
        if self.task == "segment":
            nm = 32
            nc = pred.shape[1] - 4 - nm
            cls_scores = pred[:, 4:4 + nc]
            coeff = pred[:, 4 + nc:]
        else:
            nc = pred.shape[1] - 4
            cls_scores = pred[:, 4:4 + nc]
            coeff = None
        cls = cls_scores.argmax(1)
        score = cls_scores[np.arange(cls_scores.shape[0]), cls]
        keep = score >= conf
        boxes, score, cls = boxes[keep], score[keep], cls[keep]
        coeff = coeff[keep] if coeff is not None else None
        return _xywh2xyxy(boxes), score, cls, coeff

    def _class_aware_nms(self, xyxy, score, iou):
        """NMS theo lớp (agnostic=False khớp ultralytics): offset box theo class id."""
        if xyxy.shape[0] == 0:
            return []
        max_wh = 7680.0
        return _nms(xyxy + (self._cls_off * max_wh)[:, None], score, iou)[:_MAX_DET]

    def _unletterbox(self, xyxy, r, pad_x, pad_y, w0, h0):
        xyxy = xyxy.copy()
        xyxy[:, [0, 2]] -= pad_x
        xyxy[:, [1, 3]] -= pad_y
        xyxy /= r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)
        return xyxy

    def _post_detect(self, outputs, conf, iou, r, pad_x, pad_y, w0, h0):
        xyxy, score, cls, _ = self._decode_dets(outputs[0][0], conf)
        self._cls_off = cls.astype(np.float32)
        keep = self._class_aware_nms(xyxy, score, iou)
        xyxy, score, cls = xyxy[keep], score[keep], cls[keep]
        xyxy = self._unletterbox(xyxy, r, pad_x, pad_y, w0, h0)
        return _Result(_Boxes(cls.astype(np.float32), score.astype(np.float32), xyxy))

    # ----- segmentation ----- #
    def _post_segment(self, outputs, conf, iou, r, pad_x, pad_y, w0, h0):
        # output0 = det [1, 4+nc+32, N], output1 = proto [1, 32, mh, mw]
        det, proto = outputs[0], outputs[1]
        # phân biệt output theo số chiều (export đôi khi đảo thứ tự)
        if det.ndim != 3 or proto.ndim != 4:
            det, proto = proto, det
        xyxy, score, cls, coeff = self._decode_dets(det[0], conf)
        self._cls_off = cls.astype(np.float32)
        keep = self._class_aware_nms(xyxy, score, iou)
        if len(keep) == 0:
            return _Result(_Boxes(np.zeros(0), np.zeros(0), np.zeros((0, 4))), _Masks([]))
        xyxy_lb = xyxy[keep]
        cls = cls[keep]
        score = score[keep]
        coeff = coeff[keep]

        proto = proto[0]  # [32, mh, mw]
        c, mh, mw = proto.shape
        masks = _sigmoid(coeff @ proto.reshape(c, -1)).reshape(-1, mh, mw)  # [k, mh, mw]

        sz = self._imgsz
        # crop_mask: zero ngoài box (box ở scale mask = box_lb * mh/imgsz)
        ds = xyxy_lb * (mh / sz)
        polys: List[np.ndarray] = []
        for k in range(masks.shape[0]):
            m = masks[k]
            x1, y1, x2, y2 = ds[k]
            cm = np.zeros_like(m)
            xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
            xi2, yi2 = min(mw, int(np.ceil(x2))), min(mh, int(np.ceil(y2)))
            cm[yi1:yi2, xi1:xi2] = m[yi1:yi2, xi1:xi2]
            # upsample về imgsz, threshold 0.5
            up = cv2.resize(cm, (sz, sz), interpolation=cv2.INTER_LINEAR)
            binm = (up >= 0.5).astype(np.uint8)
            cnts = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if not cnts:
                polys.append(np.zeros((0, 2), dtype=np.float32))
                continue
            big = max(cnts, key=lambda cc: cv2.contourArea(cc)).reshape(-1, 2).astype(np.float32)
            # un-letterbox polygon về ảnh gốc
            big[:, 0] = ((big[:, 0] - pad_x) / r).clip(0, w0)
            big[:, 1] = ((big[:, 1] - pad_y) / r).clip(0, h0)
            polys.append(big)

        boxes = _Boxes(cls.astype(np.float32), score.astype(np.float32),
                       self._unletterbox(xyxy_lb, r, pad_x, pad_y, w0, h0))
        return _Result(boxes, _Masks(polys))
