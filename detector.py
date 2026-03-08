import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load Model
try:
    model = YOLO("best.pt")
    print(f"✅ Model Loaded. Classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")
except Exception as e:
    print(f"❌ Error: Could not find 'best.pt' in this folder. {e}")
    exit()

# 2. Open Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow instead of MSMF
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("\n--- AMBULANCE DETECTOR ---")
print("Show ambulance images/videos to camera")
print("Press ESC to exit\n")
print("📺 Opening camera window...")

frame_count = 0

# Create window first
cv2.namedWindow("Ambulance Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ambulance Detector", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        break

    frame_count += 1

    # --- AGGRESSIVE IMAGE ENHANCEMENT ---
    enhanced_frame = frame.copy()

    # 1. Convert to LAB and apply CLAHE for contrast
    lab = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Apply histogram equalization
    enhanced_frame = cv2.convertScaleAbs(enhanced_frame)

    # 3. Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 1.0
    enhanced_frame = cv2.filter2D(enhanced_frame, -1, kernel)

    # 4. Increase brightness if image is dark
    if cv2.mean(enhanced_frame)[0] < 100:  # If too dark
        enhanced_frame = cv2.convertScaleAbs(enhanced_frame * 1.3)
    
    # Ensure frame is in valid range
    enhanced_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)

    # 3. Detection - Ultra sensitive
    results = model.predict(enhanced_frame, conf=0.02, verbose=False)

    ambulance_detected = False
    detected_boxes = []
    all_detections = []

    # Process results
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            label = model.names[class_id].lower()
            confidence = float(box.conf)

            all_detections.append(f"{label}: {confidence:.3f}")

            # Check for ambulance-related terms
            if any(keyword in label for keyword in ["ambulance", "emergency", "medical"]):
                ambulance_detected = True
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_boxes.append((x1, y1, x2, y2, label, confidence))

    # 4. Draw results on frame
    display_frame = frame.copy()

    if ambulance_detected:
        # Draw green boxes around detected ambulances
        for x1, y1, x2, y2, label, confidence in detected_boxes:
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw label with confidence
            text = f"{label.upper()}: {confidence:.2f}"
            cv2.putText(display_frame, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Large "DETECTED" banner at top
        cv2.rectangle(display_frame, (0, 0), (640, 50), (0, 255, 0), -1)
        cv2.putText(display_frame, "AMBULANCE DETECTED", (150, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        print(f"🚨 AMBULANCE DETECTED - {len(detected_boxes)} found")

    else:
        # "NOT DETECTED" message
        cv2.putText(display_frame, "NOT DETECTED", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show all detections every 30 frames for debugging
    if frame_count % 30 == 0:
        if all_detections:
            print(f"Frame {frame_count} - All detections: {all_detections}")
        else:
            print(f"Frame {frame_count} - No objects detected at all")

    # Show current confidence threshold
    cv2.putText(display_frame, "Conf: 0.02 | ENHANCED MODE", (10, display_frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 5. Display the frame with results
    cv2.imshow("Ambulance Detector", display_frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Detector stopped")
