from ultralytics import YOLO

# Load your trained model
model = YOLO("model/best.pt")

# Run detection on test images
results = model.predict(
    source="test/images",
    conf=0.5,
    save=True
)

print("Detection complete!")
