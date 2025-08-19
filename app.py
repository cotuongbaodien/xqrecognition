from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI(title="Chinese Chess Detector API")
model = YOLO("target/best.v8.pt")


def board_count(contours):
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            return approx
    return None


def detect_and_warp_board(image_bgr, dst_size=(900, 1000)):
    """
    Input: BGR image (numpy)
    Output: warped image (dst_size), M (homography), corners (src)
    Strategy:
      - Convert to gray, threshold/edge, find largest quadrilateral contour -> assume board
      - If fail, fallback: return None
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(th, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    board_cnt = board_count(sorted(contours, key=cv2.contourArea, reverse=True))
    if board_cnt is None:
        cnt = contours[0]
        x, y, wc, hc = cv2.boundingRect(cnt)
        src = np.array([[x, y], [x+wc, y], [x+wc, y+hc],[x, y+hc]], dtype="float32")
    else:
        src = board_cnt.reshape(4, 2).astype("float32")

    def order_pts(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype="float32")
    src = order_pts(src)
    dst = np.array([[0, 0], [dst_size[0]-1, 0], [dst_size[0]-1,
                   dst_size[1]-1], [0, dst_size[1]-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, dst_size)
    return warped, M, src


def map_point_to_cell(point, dst_size=(900, 1000), cols=9, rows=10):
    x, y = point
    cell_w = dst_size[0]/cols
    cell_h = dst_size[1]/rows
    col = int(x // cell_w)
    row = int(y // cell_h)
    col = max(0, min(cols-1, col))
    row = max(0, min(rows-1, row))
    cell_name = f"c{col}_r{row}"
    return {"col": col, "row": row, "cell_name": cell_name}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    np_img = np.array(img)[:, :, ::-1].copy()
    _, M, _ = detect_and_warp_board(np_img, dst_size=(900, 1000))
    results = model(np_img, imgsz=640)[0]
    out = []
    Minv = None
    if M is not None:
        Minv = np.linalg.inv(M)

    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names[cls]
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2
        cell = None
        if Minv is not None:
            pt = np.array([[cx, cy, 1.0]]).T
            warped_pt = M.dot(pt)
            warped_pt = warped_pt / warped_pt[2]
            wx, wy = float(warped_pt[0]), float(warped_pt[1])
            cell = map_point_to_cell((wx, wy), dst_size=(900, 1000), cols=9, rows=10)
        
        out.append({
            "name": name,
            "confidence": round(conf, 3),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "center": [round(cx, 2), round(cy, 2)],
            "cell": cell
        })

    return JSONResponse({"pieces": out})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
