from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="meat-1/data.yaml",
    epochs=15,
    imgsz=416,
    batch=4
)