import os
import cv2
import numpy as np

ROOT = "../dataset/pothol-yolov8"
IMG_DIR = os.path.join(ROOT, "train/images")
LBL_DIR = os.path.join(ROOT, "train/labels")
OUT_DIR = "../clean_roads"

os.makedirs(OUT_DIR, exist_ok=True)

for label_file in os.listdir(LBL_DIR):
    if not label_file.endswith(".txt"):
        continue

    base = label_file[:-4]

    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        p = os.path.join(IMG_DIR, base + ext)
        if os.path.exists(p):
            img_path = p
            break

    if img_path is None:
        continue

    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    with open(os.path.join(LBL_DIR, label_file)) as f:
        lines = f.readlines()

    for line in lines:
        _, xc, yc, bw, bh = map(float, line.split())

        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)

        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)

        x1 = max(0, x1-pad_x)
        y1 = max(0, y1-pad_y)
        x2 = min(w, x2+pad_x)
        y2 = min(h, y2+pad_y)

        mask[y1:y2, x1:x2] = 255

    cleaned = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.join(OUT_DIR, base + ".jpg"), cleaned)

print("Clean roads generated.")