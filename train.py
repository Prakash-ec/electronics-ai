from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2,
    project="runs",
    name="electronics_detector"
)

print("Training complete!")