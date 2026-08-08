from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import traceback

app = Flask(__name__)

# Allow frontend requests
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)

# ============================================================
# PATH CONFIGURATION
# ============================================================

# backend/app.py
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# D:\electronics-ai
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

# D:\electronics-ai\model\best.pt
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "model",
    "best.pt"
)

print("=" * 60)
print("ELECTRONICS AI BACKEND")
print("=" * 60)

print("Backend directory:")
print(BACKEND_DIR)

print("Project directory:")
print(PROJECT_DIR)

print("Model path:")
print(MODEL_PATH)

print("Model exists:", os.path.exists(MODEL_PATH))

# ============================================================
# LOAD YOLO MODEL
# ============================================================

model = None
model_error = None

try:

    if not os.path.exists(MODEL_PATH):

        model_error = f"Model file not found: {MODEL_PATH}"

        print("ERROR:", model_error)

    else:

        print("Loading YOLO model...")

        model = YOLO(MODEL_PATH)

        print("YOLO model loaded successfully!")

        print("Classes:")

        print(model.names)

except Exception as e:

    model_error = str(e)

    print("FAILED TO LOAD YOLO MODEL")
    print(model_error)

    traceback.print_exc()


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "success": True,
        "message": "Electronics AI Backend Running",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_error": model_error
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    if model is None:

        return jsonify({
            "success": False,
            "status": "error",
            "message": "YOLO model is not loaded",
            "model_exists": os.path.exists(MODEL_PATH),
            "model_path": MODEL_PATH,
            "error": model_error
        }), 500

    return jsonify({
        "success": True,
        "status": "healthy",
        "message": "YOLO model is loaded",
        "model_loaded": True,
        "classes": model.names
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    print("=" * 60)
    print("Prediction request received")

    # Check model
    if model is None:

        print("ERROR: YOLO model is not loaded")

        return jsonify({
            "success": False,
            "error": "YOLO model is not loaded",
            "details": model_error
        }), 503

    # Check image
    if "image" not in request.files:

        print("ERROR: No image received")

        return jsonify({
            "success": False,
            "error": "No image received"
        }), 400

    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        file = request.files["image"]

        print("Received file:", file.filename)

        image_bytes = file.read()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "Empty image"
            }), 400

        # ----------------------------------------------------
        # Convert bytes to OpenCV image
        # ----------------------------------------------------

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
                "error": "Invalid image"
            }), 400

        print(
            "Image received:",
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

        result = results[0]

        detections = []

        # ----------------------------------------------------
        # Extract detections
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = model.names[
                    class_id
                ]

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

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "detections": detections,

            "count": len(detections)

        })

    except Exception as e:

        print("=" * 60)
        print("PREDICTION ERROR")
        print(str(e))
        traceback.print_exc()
        print("=" * 60)

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

    print("=" * 60)
    print(f"Starting Flask server on port {port}")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )