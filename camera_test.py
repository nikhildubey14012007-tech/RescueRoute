import cv2
from ultralytics import YOLO

# 1. Load Model
model = YOLO("best.pt")

# 2. Setup Camera (tries 0, 1, or 2)
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Connected to Camera {i}")
        break

if not cap or not cap.isOpened():
    print("❌ No camera found!")
    exit()

print("\n--- INSTRUCTIONS ---")
print("1. Showing a Google Image? Tilt the screen slightly to avoid light glare.")
print("2. If nothing is found, check the console for 'Detected' list.")
print("3. Press ESC to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Run detection with lower confidence to see 'weak' matches
    results = model.predict(frame, conf=0.2, verbose=False)
    
    # Process and Draw
    for r in results:
        annotated_frame = r.plot() # Draws boxes automatically
        
        for box in r.boxes:
            name = model.names[int(box.cls)]
            if "ambulance" in name.lower():
                print(f"🚨 AMBULANCE DETECTED! Confidence: {float(box.conf):.2f}")
                cv2.putText(annotated_frame, "!!! AMBULANCE !!!", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("Ambulance Detection - Live View", annotated_frame)
    
    if cv2.waitKey(1) == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()