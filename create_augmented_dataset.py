import cv2
import os
import random
import numpy as np

BG_DIR = "../clean_roads"
REAL_DIR = "../pothole_crops/pothole"
GAN_DIR = "../synthetic_potholes"

OUT_IMG = "../augmented_dataset/images"
OUT_LBL = "../augmented_dataset/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)


# ---------------- HELPERS ----------------
def get_valid_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]


bg_files = get_valid_images(BG_DIR)
real_files = get_valid_images(REAL_DIR)
gan_files = get_valid_images(GAN_DIR)


# ---------------- BLENDING ----------------
def blend_patch(bg, patch, x, y):
    h, w = patch.shape[:2]

    # Irregular mask
    mask = np.zeros((h, w), dtype=np.float32)

    pts = np.array([
        [random.randint(0, w//4), random.randint(0, h//4)],
        [random.randint(3*w//4, w-1), random.randint(0, h//4)],
        [random.randint(3*w//4, w-1), random.randint(3*h//4, h-1)],
        [random.randint(0, w//4), random.randint(3*h//4, h-1)]
    ])

    cv2.fillPoly(mask, [pts], 1)

    # Softer edges (realistic)
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    mask = np.expand_dims(mask, axis=2)

    roi = bg[y:y+h, x:x+w].astype(np.float32)
    patch = patch.astype(np.float32)

    # Brightness matching
    patch *= np.mean(roi) / (np.mean(patch) + 1e-5)

    # Edge darkening (important for realism)
    edges = cv2.Canny(patch.astype(np.uint8), 50, 150)
    if edges is not None and edges.size > 0:
        edges = edges.astype(np.float32)
        edges = np.expand_dims(edges, axis=2)

        patch = patch - edges * 0.2
        patch = np.clip(patch, 0, 255)

    # Blend
    blended = roi * (1 - mask) + patch * mask
    bg[y:y+h, x:x+w] = blended.astype(np.uint8)

    return bg


# ---------------- MAIN LOOP ----------------
for i in range(500):

    bg_path = os.path.join(BG_DIR, random.choice(bg_files))
    bg = cv2.imread(bg_path)

    if bg is None:
        continue

    H, W = bg.shape[:2]
    out_name = f"aug_{i}.jpg"

    label_path = os.path.join(OUT_LBL, out_name.replace(".jpg", ".txt"))
    open(label_path, "w").close()

    num_potholes = random.randint(1, 2)

    for _ in range(num_potholes):

        # 90% real, 10% GAN
        if random.random() < 0.9:
            p_path = os.path.join(REAL_DIR, random.choice(real_files))
        else:
            if len(gan_files) == 0:
                continue
            p_path = os.path.join(GAN_DIR, random.choice(gan_files))

        p = cv2.imread(p_path)

        if p is None or len(p.shape) != 3:
            continue

        # Controlled scaling
        scale = random.uniform(1.2, 2.5)
        p = cv2.resize(p, None, fx=scale, fy=scale)

        # Slight blur (optional)
        if random.random() < 0.4:
            p = cv2.GaussianBlur(p, (3, 3), 0)

        # Lighting variation
        p = (p * random.uniform(0.75, 1.0)).astype(np.uint8)

        h, w = p.shape[:2]

        if h >= H or w >= W:
            continue

        # X placement
        if W - w - 1 <= 0:
            continue
        x = random.randint(0, W - w - 1)

        # Y placement (bottom half)
        y_min = int(H * 0.5)
        y_max = H - h - 1

        if y_max <= y_min:
            continue

        y = random.randint(y_min, y_max)

        # Blend pothole
        bg = blend_patch(bg, p, x, y)

        # YOLO label
        xc = (x + w / 2) / W
        yc = (y + h / 2) / H
        bw = w / W
        bh = h / H

        with open(label_path, "a") as f:
            f.write(f"0 {xc} {yc} {bw} {bh}\n")

    cv2.imwrite(os.path.join(OUT_IMG, out_name), bg)

print(" FINAL realistic dataset created successfully!")