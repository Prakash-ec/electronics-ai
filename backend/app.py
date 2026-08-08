from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained YOLO model
model = YOLO("../model/best.pt")


@app.route("/")
def home():
    return "Electronics AI Backend Running"


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    file = request.files["image"]

    image_bytes = file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model.predict(
        source=image,
        conf=0.40,
        verbose=False
    )

    detections = []

    result = results[0]

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        detections.append({
            "class": class_name,
            "confidence": round(confidence * 100, 2),
            "box": [x1, y1, x2, y2]
        })

    return jsonify({
        "detections": detections
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )