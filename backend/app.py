from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

# --------------------------------------------------
# Flask app
# --------------------------------------------------

app = Flask(__name__)

# Allow frontend requests from local computer, Netlify, etc.
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)

# --------------------------------------------------
# Find project directory
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Your project structure should be:
#
# electronics-ai/
# │
# ├── app.py
# ├── requirements.txt
# │
# ├── model/
# │   └── best.pt
# │
# └── frontend/
#     └── index.html
#

MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")

print("=" * 50)
print("Electronics AI Backend")
print("=" * 50)
print("Base directory:", BASE_DIR)
print("Model path:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

try:
    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully!")

except Exception as e:
    print("ERROR loading YOLO model:")
    print(str(e))
    model = None


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "success": True,
        "message": "Electronics AI Backend Running",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "healthy",
        "model_loaded": model is not None
    })


# --------------------------------------------------
# Prediction route
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check model
    if model is None:

        return jsonify({
            "success": False,
            "error": "YOLO model is not loaded"
        }), 500

    # Check image
    if "image" not in request.files:

        return jsonify({
            "success": False,
            "error": "No image received. Use form field name 'image'."
        }), 400

    try:

        # --------------------------------------------------
        # Get uploaded image
        # --------------------------------------------------

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        print("Received image:", file.filename)

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

        image_bytes = file.read()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "Uploaded image is empty"
            }), 400

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({
                "success": False,
                "error": "Invalid image format"
            }), 400

        print(
            "Image received:",
            image.shape
        )

        # --------------------------------------------------
        # YOLO prediction
        # --------------------------------------------------

        results = model.predict(
            source=image,
            conf=0.40,
            verbose=False
        )

        result = results[0]

        detections = []

        # --------------------------------------------------
        # Extract detections
        # --------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
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

        print(
            "Detections:",
            detections
        )

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return jsonify({

            "success": True,

            "count": len(detections),

            "detections": detections

        })

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# --------------------------------------------------
# Run locally
# --------------------------------------------------

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