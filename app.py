from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load your model (ensure best.pt is in the same folder)
try:
    model = YOLO("best.pt")
    print(f"✅ Model loaded. Classes found: {model.names}")
except Exception as e:
    print(f"❌ Error loading best.pt: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Online", "message": "Ambulance Detection API", "classes": model.names})

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # conf=0.1 helps catch objects even if the camera quality is poor
    results = model.predict(img, conf=0.1, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            conf = float(box.conf)
            
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "is_ambulance": "ambulance" in label.lower()
            })

    return jsonify({
        "total_detections": len(detections),
        "results": detections
    })

if __name__ == "__main__":
    # Host 0.0.0.0 makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)