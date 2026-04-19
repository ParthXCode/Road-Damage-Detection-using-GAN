import os
import shutil

# Original dataset
REAL_IMG = "../dataset/pothol-yolov8/train/images"
REAL_LBL = "../dataset/pothol-yolov8/train/labels"

# Synthetic augmented dataset
AUG_IMG = "../augmented_dataset/images"
AUG_LBL = "../augmented_dataset/labels"

# Final merged dataset
FINAL_ROOT = "../final_dataset"
FINAL_IMG = os.path.join(FINAL_ROOT, "train/images")
FINAL_LBL = os.path.join(FINAL_ROOT, "train/labels")

os.makedirs(FINAL_IMG, exist_ok=True)
os.makedirs(FINAL_LBL, exist_ok=True)

# Copy real + synthetic images
for folder in [REAL_IMG, AUG_IMG]:
    for file in os.listdir(folder):
        shutil.copy(
            os.path.join(folder, file),
            os.path.join(FINAL_IMG, file)
        )

# Copy real + synthetic labels
for folder in [REAL_LBL, AUG_LBL]:
    for file in os.listdir(folder):
        shutil.copy(
            os.path.join(folder, file),
            os.path.join(FINAL_LBL, file)
        )

# Auto-create data.yaml
yaml_content = """path: ../final_dataset
train: train/images
val: ../dataset/pothol-yolov8/valid/images

nc: 1
names: ['pothole']
"""

with open(os.path.join(FINAL_ROOT, "data.yaml"), "w") as f:
    f.write(yaml_content)

print("Merged dataset created successfully.")
print("data.yaml auto-generated in final_dataset/")