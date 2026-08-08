from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load trained YOLO model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully")
print("Model:", MODEL_PATH)


# -----------------------------
# Home
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Electronics AI Backend Running"


# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image received"
        }), 400

    file = request.files["image"]

    try:
        # Read uploaded image
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
            return jsonify({
                "error": "Invalid image"
            }), 400

        # Run YOLO
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
                "confidence": round(
                    confidence * 100,
                    2
                ),
                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            })

        return jsonify({
            "success": True,
            "detections": detections
        })

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -----------------------------
# Start server
# -----------------------------
if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )