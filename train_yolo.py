from ultralytics import YOLO


def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="../final_dataset/data.yaml",
        epochs=100,
        imgsz=832,
        batch=8,
        device=0,
        workers=2
    )


if __name__ == "__main__":
    main()