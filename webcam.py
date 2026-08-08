from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("model/best.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

print("Camera started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera")
        break

    # Run YOLO detection
    results = model.predict(
        source=frame,
        conf=0.60,
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("Electronics Component Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()