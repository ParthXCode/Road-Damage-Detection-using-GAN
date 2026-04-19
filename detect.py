from ultralytics import YOLO

model = YOLO(r"C:\Users\manas\runs\detect\train2\weights\best.pt")

results = model(
    source="../test_road.jpg",
    show=True,
    save=True,
    conf=0.25
)