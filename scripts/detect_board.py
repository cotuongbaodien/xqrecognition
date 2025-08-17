import cv2
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent.parent

BOARD_IMG = f"{base_dir}/scripts/pieces/board.jpg"
img = cv2.imread(BOARD_IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:  # tìm được tứ giác
        corners = approx.reshape(4, 2)
        print("Board corners:", corners)
        break
