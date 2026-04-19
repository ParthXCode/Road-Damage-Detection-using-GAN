import os
import cv2

ROOT = "../dataset/pothol-yolov8"
IMG_DIR = os.path.join(ROOT, "train/images")
LBL_DIR = os.path.join(ROOT, "train/labels")
OUT_DIR = "../pothole_crops/pothole"

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

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(os.path.join(LBL_DIR, label_file)) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        _, xc, yc, bw, bh = map(float, line.split())

        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)

        crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

        if crop.size > 0:
            cv2.imwrite(f"{OUT_DIR}/{base}_{i}.png", crop)

print("Pothole crops extracted.")