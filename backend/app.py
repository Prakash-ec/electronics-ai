```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import traceback

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# ============================================================
# CORS CONFIGURATION
# ============================================================

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    },
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)


# Extra CORS headers for every response
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# ============================================================
# PROJECT PATH
# ============================================================

# app.py is inside:
#
# electronics-ai/
# ├── backend/
# │   └── app.py
# ├── model/
# │   └── best.pt
# └── frontend/
#     └── index.html

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

print("========================================")
print("Electronics AI Backend Starting")
print("========================================")
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))


# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = None
model_error = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully!")
    print("Classes:", model.names)

except Exception as e:
    model_error = str(e)

    print("========================================")
    print("YOLO MODEL LOAD ERROR")
    print("========================================")
    print(model_error)

    traceback.print_exc()


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "status": "healthy",
        "message": "Electronics AI Backend Running",
        "model_loaded": model is not None
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    if model is not None:
        return jsonify({
            "success": True,
            "status": "healthy",
            "model_loaded": True,
            "message": "YOLO model is loaded",
            "classes": model.names
        })

    return jsonify({
        "success": False,
        "status": "unhealthy",
        "model_loaded": False,
        "message": "YOLO model is not loaded",
        "error": model_error
    }), 500


# ============================================================
# OPTIONS / CORS PREFLIGHT
# ============================================================

@app.route("/predict", methods=["OPTIONS"])
def predict_options():

    response = jsonify({
        "success": True
    })

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"

    return response


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    print("========================================")
    print("Prediction request received")
    print("========================================")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None:

        print("ERROR: YOLO model is not loaded")

        return jsonify({
            "success": False,
            "error": "YOLO model is not loaded"
        }), 500

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "image" not in request.files:

        print("ERROR: No image received")

        return jsonify({
            "success": False,
            "error": "No image received"
        }), 400

    try:

        # ----------------------------------------------------
        # Get uploaded file
        # ----------------------------------------------------

        file = request.files["image"]

        print("Image filename:", file.filename)

        # ----------------------------------------------------
        # Read image bytes
        # ----------------------------------------------------

        image_bytes = file.read()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "Empty image file"
            }), 400

        print(
            "Image size:",
            len(image_bytes),
            "bytes"
        )

        # ----------------------------------------------------
        # Convert bytes -> NumPy
        # ----------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # Decode image
        # ----------------------------------------------------

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            print("ERROR: Could not decode image")

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        print(
            "Image decoded:",
            image.shape
        )

        # ----------------------------------------------------
        # YOLO prediction
        # ----------------------------------------------------

        print("Running YOLO prediction...")

        results = model.predict(
            source=image,
            conf=0.40,
            verbose=False
        )

        # ----------------------------------------------------
        # Get result
        # ----------------------------------------------------

        result = results[0]

        detections = []

        # ----------------------------------------------------
        # Process bounding boxes
        # ----------------------------------------------------

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

        print(
            "Detections:",
            detections
        )

        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "message": "Prediction successful",

            "detections": detections,

            "count": len(detections)

        })

    except Exception as e:

        print("========================================")
        print("PREDICTION ERROR")
        print("========================================")

        print(str(e))

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        f"Starting Flask server on port {port}"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

