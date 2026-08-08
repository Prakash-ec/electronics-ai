import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np


app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# Find best.pt reliably
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best.pt"
)


# Load trained YOLO model
print("Loading YOLO model...")
print("Model path:", MODEL_PATH)

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")


# --------------------------------------------------
# Home / Health check
# --------------------------------------------------

@app.route("/")
def home():
    return "Electronics AI Backend Running"


# --------------------------------------------------
# YOLO Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether image was received
    if "image" not in request.files:
        return jsonify({
            "error": "No image received"
        }), 400


    # Get uploaded image
    file = request.files["image"]


    # Read image bytes
    image_bytes = file.read()


    # Convert bytes to NumPy array
    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )


    # Decode image
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    # Check image
    if image is None:
        return jsonify({
            "error": "Invalid image"
        }), 400


    # --------------------------------------------------
    # Run YOLO
    # --------------------------------------------------

    results = model.predict(
        source=image,
        conf=0.40,
        verbose=False
    )


    detections = []


    result = results[0]


    # --------------------------------------------------
    # Process detections
    # --------------------------------------------------

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_name = model.names[
            class_id
        ]


        # Bounding box
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


    # --------------------------------------------------
    # Return JSON
    # --------------------------------------------------

    return jsonify({

        "detections": detections

    })


# --------------------------------------------------
# Start Flask server
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )